"""Crop-vs-gallery matching, multi-frame aggregation, and receipt reconciliation.
Each crop is scored only against the receipt SKUs; per-track votes are aggregated,
then a bipartite assignment reconciles visible items against paid quantities."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cartgate import config


# ---------- similarity ----------

def best_sim_against_sku(vec: np.ndarray, gallery_entry: dict) -> float:
    """Max cosine similarity against all gallery variants of one SKU."""
    return float(gallery_entry["vectors"] @ vec)  # vectors are L2-normalized


def sku_similarity(vec: np.ndarray, gallery: dict, sku: str) -> float:
    return float(np.max(gallery[sku]["vectors"] @ vec))


def sift_inliers(query_bgr: np.ndarray, ref_bgr: np.ndarray, max_side: int = 320) -> int:
    """Geometric verification: SIFT matches consistent with a homography."""
    def prep(img):
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=800)
    k1, d1 = sift.detectAndCompute(prep(query_bgr), None)
    k2, d2 = sift.detectAndCompute(prep(ref_bgr), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    bf = cv2.BFMatcher()
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(inl.sum()) if inl is not None else 0


# ---------- per-frame matching ----------

@dataclass
class TrackEvidence:
    votes: Counter = field(default_factory=Counter)   # sku -> weighted votes
    sims: dict = field(default_factory=lambda: defaultdict(list))
    n_frames: int = 0


def match_frame_crops(frame, embedder, gallery: dict, receipt_skus: list[str],
                      sim_threshold: float) -> list[dict]:
    """Embed each detected crop; score against receipt SKUs (candidate set)."""
    results = []
    for obj in frame.objects:
        x0, y0, x1, y1 = obj.box
        crop = frame.image[y0:y1 + 1, x0:x1 + 1]
        if crop.size == 0:
            continue
        vec = embedder.embed(crop, None)
        scored = sorted(((sku_similarity(vec, gallery, s), s) for s in set(receipt_skus)),
                        reverse=True)
        best_sim, best_sku = scored[0]
        results.append({
            "track_id": obj.track_id,
            "true_sku": obj.sku,           # kept for evaluation only
            "pred_sku": best_sku if best_sim >= sim_threshold else None,
            "sim": best_sim,
            "crop": crop,
        })
    return results


def aggregate_tracks(per_frame_results: list[list[dict]]) -> dict[int, TrackEvidence]:
    tracks: dict[int, TrackEvidence] = defaultdict(TrackEvidence)
    for frame_results in per_frame_results:
        for r in frame_results:
            t = tracks[r["track_id"]]
            t.n_frames += 1
            if r["pred_sku"] is not None:
                t.votes[r["pred_sku"]] += r["sim"]
                t.sims[r["pred_sku"]].append(r["sim"])
    return tracks


# ---------- receipt verification ----------

def verify_cart(tracks: dict, receipt: dict[str, int], sim_threshold: float,
                unexplained_tolerance: int = 0) -> dict:
    """receipt: {sku: paid_quantity}. Returns verdict dict."""
    # Resolve each track to its majority SKU (or None = unexplained)
    resolved = {}
    for tid, ev in tracks.items():
        if not ev.votes:
            resolved[tid] = (None, 0.0)
            continue
        sku, _ = ev.votes.most_common(1)[0]
        resolved[tid] = (sku, float(np.mean(ev.sims[sku])))

    # Bipartite assignment: tracks x receipt slots (each unit of qty = one slot)
    slots = [s for s, q in receipt.items() for _ in range(q)]
    tids = list(resolved.keys())
    if slots and tids:
        cost = np.ones((len(tids), len(slots)), np.float32)
        for i, tid in enumerate(tids):
            sku, sim = resolved[tid]
            for j, slot_sku in enumerate(slots):
                if sku == slot_sku:
                    cost[i, j] = 1.0 - sim
        ri, ci = linear_sum_assignment(cost)
        assigned = {tids[i]: slots[j] for i, j in zip(ri, ci) if cost[i, j] < 1.0 - sim_threshold * 0.5}
    else:
        assigned = {}

    unexplained = [tid for tid in tids if tid not in assigned]
    used_slots = Counter(assigned.values())
    unseen = {s: q - used_slots.get(s, 0) for s, q in receipt.items() if q > used_slots.get(s, 0)}

    flags = []
    for tid in unexplained:
        sku, sim = resolved[tid]
        if sku is None:
            flags.append({"type": "UNRECOGNIZED_ITEM", "track": tid,
                          "detail": "visible item matches nothing on the receipt"})
        else:
            flags.append({"type": "QUANTITY_EXCEEDED", "track": tid, "sku": sku,
                          "detail": f"extra '{sku}' beyond paid quantity"})

    verdict = "PASS" if len(flags) <= unexplained_tolerance else "FLAG"
    return {
        "verdict": verdict,
        "flags": flags,
        "matched": {tid: assigned[tid] for tid in assigned},
        "unseen_paid_items": unseen,   # informational: occlusion-tolerant
        "resolved": {tid: {"sku": s, "sim": round(sim, 3)} for tid, (s, sim) in resolved.items()},
    }


# ---------- 3-way decision (PASS / FLAG / REVIEW) ----------

def _resolve(tracks: dict, pass_sim: float, review_sim: float, min_frames: int) -> list[dict]:
    """Collapse each multi-frame track to one record with a confidence band:
    strong (sim>=pass_sim), weak (>=review_sim), or none. A track seen in fewer
    than min_frames frames is 'unstable' and never used to accuse."""
    out = []
    for tid, ev in tracks.items():
        stable = ev.n_frames >= min_frames
        if not ev.votes:
            out.append({"tid": tid, "sku": None, "sim": 0.0, "band": "none",
                        "frames": ev.n_frames, "stable": stable})
            continue
        sku, _ = ev.votes.most_common(1)[0]
        sim = float(np.mean(ev.sims[sku]))
        band = "strong" if sim >= pass_sim else ("weak" if sim >= review_sim else "none")
        out.append({"tid": tid, "sku": sku if band != "none" else None, "sim": sim,
                    "band": band, "frames": ev.n_frames, "stable": stable})
    return out


def decide_cart(tracks: dict, receipt: dict[str, int], *,
                pass_sim: float = config.PASS_SIM, review_sim: float = config.REVIEW_SIM,
                min_frames: int = config.MIN_FRAMES) -> dict:
    """Receipt-conditioned, occlusion-tolerant exit-gate decision -> PASS/FLAG/REVIEW.

      * Paid-but-unseen items are expected (can't see a full cart) -> never penalized.
      * A stable, confident item matching nothing on the receipt   -> FLAG (unpaid).
      * More confident instances of a SKU than were paid for       -> FLAG (over-qty).
      * Weak / unstable evidence                                   -> REVIEW (human).
      * Otherwise                                                  -> PASS.

    Thresholds are per-deployment and must be calibrated on real gate footage.
    """
    resolved = _resolve(tracks, pass_sim, review_sim, min_frames)

    # Assign only STRONG + STABLE tracks to paid receipt slots (1 slot per unit qty).
    slots = [s for s, q in receipt.items() for _ in range(q)]
    strong = [r for r in resolved if r["band"] == "strong" and r["stable"]]
    assigned_tids, used = set(), Counter()
    if slots and strong:
        cost = np.ones((len(strong), len(slots)), np.float32)
        for i, r in enumerate(strong):
            for j, slot in enumerate(slots):
                if r["sku"] == slot:
                    cost[i, j] = 1.0 - r["sim"]
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] < 1.0:                       # sku actually matched this slot
                assigned_tids.add(strong[i]["tid"])
                used[slots[j]] += 1

    flags, reviews = [], []
    for r in resolved:
        if r["tid"] in assigned_tids:
            continue
        if not r["stable"]:                            # fleeting detection -> never accuse
            if r["band"] in ("strong", "weak"):
                reviews.append({**r, "reason": "unstable track (few frames), verify manually"})
            continue
        if r["band"] == "strong":                      # confident extra of a paid SKU
            flags.append({"type": "QUANTITY_EXCEEDED", "sku": r["sku"], "track": r["tid"],
                          "sim": round(r["sim"], 3),
                          "detail": f"more '{r['sku']}' seen than paid for"})
        elif r["band"] == "weak":                      # not sure what it is
            reviews.append({**r, "reason": "ambiguous match to receipt, verify manually"})
        else:                                          # band none = matches nothing paid
            flags.append({"type": "UNRECOGNIZED_ITEM", "track": r["tid"],
                          "sim": round(r["sim"], 3),
                          "detail": "stable visible item matches no paid receipt line"})

    used_by_sku = used
    unseen = {s: q - used_by_sku.get(s, 0) for s, q in receipt.items()
              if q > used_by_sku.get(s, 0)}           # occlusion-tolerant: info only

    if flags:
        verdict = "FLAG"
    elif reviews:
        verdict = "REVIEW"
    else:
        verdict = "PASS"

    n_visible = sum(1 for r in resolved if r["stable"])
    reasons = [f["detail"] for f in flags] + [r["reason"] for r in reviews]
    if verdict == "PASS":
        reasons = [f"{len(used_by_sku)} paid item(s) confirmed; "
                   f"{sum(unseen.values())} paid item(s) unseen (occlusion, tolerated)"]

    return {
        "verdict": verdict,                            # PASS | FLAG | REVIEW
        "reasons": reasons,
        "flags": flags,                                # actionable: likely unpaid / over-qty
        "review_items": reviews,                       # ambiguous: send to admin
        "counts": {"visible_stable": n_visible, "paid_total": sum(receipt.values()),
                   "confirmed_paid": int(sum(used_by_sku.values()))},
        "unseen_paid_items": unseen,                   # occlusion-tolerant
        "resolved": resolved,
    }
