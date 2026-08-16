"""End-to-end demo: detect items in each camera's frames, recognize them against
the cart's receipt, fuse the 2 cameras, and return a PASS/FLAG/REVIEW verdict.

Every per-object piece of evidence a downstream decision layer could need
(track id, camera id, boxes, frame count, detector confidence and the FULL
candidate-similarity vector, not just the argmax) is preserved through the
pipeline and serialized by --out.

Counting identical duplicates across cameras really needs camera geometry;
max-over-cameras is a calibratable stand-in until that lands.
"""
import argparse
import datetime as dt
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cartgate import config
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_frames
from cartgate.match import sku_similarity

SCHEMA_VERSION = "1.1"
FUSION_STRATEGY = "legacy_max_count"   # replaced once cartgate.fusion lands


def load_cutouts(cut_dir: str) -> dict:
    cut = defaultdict(list)
    for p in sorted(Path(cut_dir).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_camera(model, frames, embedder, gallery, receipt_skus, dev,
                   camera_id: str = "cam0") -> list[dict]:
    """Detect over one camera's frames and resolve each object to a receipt SKU.

    Returns one record per tracked object, keeping all evidence:
      instance_id, track_ids, camera_ids, candidates (similarity against EVERY
      receipt SKU, which a global assignment needs), best_sku, sim, band,
      n_frames, stable, boxes, det_conf.

    Candidates are restricted to the cart's receipt SKUs by design — that is the
    receipt-conditioned contract, not a shortcut.

    Object identity across frames uses IoU association to the synthetic GT as a
    stand-in for ByteTrack (which would track from real 30fps video).
    """
    rc = sorted(set(str(s) for s in receipt_skus))
    tracks = defaultdict(lambda: {"votes": Counter(), "win_sims": defaultdict(list),
                                  "cand_sums": defaultdict(float), "n": 0,
                                  "boxes": [], "confs": []})
    for f in frames:
        res = model.predict(f.image, conf=config.DET_CONF, verbose=False, device=dev)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for box, det_conf in zip(xyxy, confs):
            bx = tuple(int(v) for v in box)
            oid, best = None, config.TRACK_IOU      # associate detection -> GT object id
            for o in f.objects:
                j = _iou(bx, o.box)
                if j > best:
                    best, oid = j, o.track_id
            if oid is None:
                continue                             # spurious detection (no GT) -> skip
            crop = f.image[max(0, bx[1]):bx[3], max(0, bx[0]):bx[2]]
            if crop.size == 0:
                continue
            vec = embedder.embed(crop, None)
            sims = {s: sku_similarity(vec, gallery, s) for s in rc}   # keep ALL, not argmax
            sku, sim = (max(sims.items(), key=lambda kv: kv[1]) if sims else (None, 0.0))
            t = tracks[oid]
            t["n"] += 1
            t["votes"][sku] += 1
            t["win_sims"][sku].append(sim)
            for s, v in sims.items():
                t["cand_sums"][s] += v
            t["boxes"].append([int(v) for v in bx])
            t["confs"].append(float(det_conf))

    resolved = []
    for tid, t in sorted(tracks.items(), key=lambda kv: str(kv[0])):
        if not t["votes"]:
            continue
        sku = t["votes"].most_common(1)[0][0]
        sim = float(np.mean(t["win_sims"][sku]))
        b = config.band(sim)
        resolved.append({
            "instance_id": f"{camera_id}#{tid}",
            "track_ids": [int(tid)],
            "camera_ids": [camera_id],
            "plane_xy": None,                        # needs calibration + plane fusion
            "candidates": {s: round(v / t["n"], 4) for s, v in sorted(t["cand_sums"].items())},
            "best_sku": str(sku) if b != "none" else None,
            "sim": round(sim, 3),
            "band": b,
            "n_frames": int(t["n"]),
            "stable": bool(t["n"] >= config.MIN_FRAMES),
            "boxes": t["boxes"],
            "det_conf": round(float(np.mean(t["confs"])), 3),
            "label_conflict": False,                 # filled by _mark_label_conflicts
        })
    return resolved


def _mark_label_conflicts(per_cam: list[list[dict]], receipt: dict) -> None:
    """PROVISIONAL: flag instances that collide on one receipt line.

    Within a single camera, if more confident instances resolve to a SKU than were
    paid for, at least one of them is mislabeled — that is what a global assignment
    is supposed to disentangle. Replace this with cartgate.fusion's own definition
    once that module is available.
    """
    for cam in per_cam:
        counts = Counter(r["best_sku"] for r in cam
                         if r["best_sku"] and r["band"] == "strong" and r["stable"])
        for r in cam:
            paid = int(receipt.get(r["best_sku"], 0)) if r["best_sku"] else 0
            r["label_conflict"] = bool(r["best_sku"] and counts[r["best_sku"]] > paid)


def fuse_and_decide(per_cam: list[list[dict]], receipt: dict) -> dict:
    """Fuse the cameras (max-count per SKU) and return the 3-way verdict."""
    receipt = {str(s): int(q) for s, q in receipt.items()}
    _mark_label_conflicts(per_cam, receipt)

    def maxcount(pred):
        return max((sum(1 for r in cam if pred(r)) for cam in per_cam), default=0)

    seen_skus = set(receipt) | {r["best_sku"] for cam in per_cam for r in cam if r["best_sku"]}
    fused = {s: maxcount(lambda r, s=s: r["best_sku"] == s and r["band"] == "strong" and r["stable"])
             for s in seen_skus}
    unrec = maxcount(lambda r: r["band"] == "none" and r["stable"])   # matches nothing paid
    weak = maxcount(lambda r: r["band"] == "weak" and r["stable"])    # ambiguous

    flags, reviews = [], []
    excess = {}
    for s, paid in receipt.items():
        if fused.get(s, 0) > paid:
            excess[s] = fused[s] - paid
            flags.append({"type": "QUANTITY_EXCEEDED", "sku": s,
                          "detail": f"'{s}' seen {fused[s]} > paid {paid}"})
    if unrec > 0:
        flags.append({"type": "UNRECOGNIZED_ITEM",
                      "detail": f"{unrec} visible item(s) match no paid receipt line"})
    if weak > 0:
        reviews.append({"type": "AMBIGUOUS", "detail": f"{weak} item(s) ambiguous, verify"})

    unseen = {s: paid - fused.get(s, 0) for s, paid in receipt.items() if fused.get(s, 0) < paid}
    verdict = "FLAG" if flags else ("REVIEW" if reviews else "PASS")
    return {"verdict": verdict, "flags": flags, "review_items": reviews,
            "fused_counts": {s: c for s, c in fused.items() if c},
            "excess_counts": excess,
            "unseen_paid_items": unseen,
            "fusion_strategy": FUSION_STRATEGY}


def build_observation(transaction_id: str, per_cam: list[list[dict]], decision: dict,
                      frames_per_cam: int, duration_ms: float, camera_ids: list[str]) -> dict:
    """Serialize one cart into the vision->decision contract (schema v1.1)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "gate_id": config.GATE_ID,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "duration_ms": round(float(duration_ms), 1),
        "vision_status": "VALID",
        "cameras": [{"camera_id": cid, "status": "OK", "frames_used": int(frames_per_cam)}
                    for cid in camera_ids],
        "instances": [r for cam in per_cam for r in cam],
        "fused_counts": decision["fused_counts"],
        "excess_counts": decision["excess_counts"],
        "fusion_strategy": decision["fusion_strategy"],
    }


def run(dataset, cutouts_dir, weights, onnx, dev, seed=7, out_path=None, n_frames=4):
    rng = np.random.default_rng(seed)
    embedder = get_embedder(onnx, pad=("dino" in onnx or "vit" in onnx))
    print(f"[1/4] embedder: {embedder.name}  (pad={embedder.pad}, "
          f"providers={getattr(embedder, 'providers', ['classical'])})")
    print("[2/4] building gallery (studio + synthetic enriched views)...")
    gallery = build_gallery(dataset, embedder, "out", remove_bg=False, enrich_synth=16)
    cutouts = load_cutouts(cutouts_dir)
    model = YOLO(weights)
    print(f"[3/4] detector: {weights}  (device={dev})")

    skus = sorted(cutouts.keys())
    base = [str(s) for s in rng.choice(skus, size=3, replace=False)]   # str: keeps JSON serializable
    extra = next(s for s in skus if s not in base)
    scenarios = [
        ("정상 결제 카트", list(base), {s: 1 for s in base}, "PASS"),
        ("미결제 물건 포함", base + [extra], {s: 1 for s in base}, "FLAG"),
        ("수량 초과(1결제 2적재)", base + [base[0]], {s: 1 for s in base}, "FLAG"),
        ("결제했지만 가려짐", base[:2], {s: 1 for s in base}, "PASS"),
    ]
    cams = [(-1.0, "cam_left"), (1.0, "cam_right")]      # 2 upper-diagonal cameras
    print("[4/4] gate scenarios (real detection -> 2-camera fusion -> decision)\n")
    results, observations = [], []
    for i, (name, cart, receipt, expect) in enumerate(scenarios):
        # frames stand in for camera capture -> generated OUTSIDE the timer, so
        # duration_ms measures only detect+recognize+fuse (what a gate would spend).
        cam_frames = [(cam_id, synth_cart_frames(cart, cutouts, rng, n_frames=n_frames,
                                                 size=(640, 640), cam_dirs=[d] * n_frames))
                      for d, cam_id in cams]
        t0 = time.perf_counter()
        per_cam = [resolve_camera(model, frames, embedder, gallery,
                                  list(receipt.keys()), dev, camera_id=cam_id)
                   for cam_id, frames in cam_frames]
        decision = fuse_and_decide(per_cam, receipt)
        duration_ms = (time.perf_counter() - t0) * 1000
        ok = decision["verdict"] == expect
        results.append((name, expect, decision, ok))
        observations.append(build_observation(f"demo-{i:02d}", per_cam, decision,
                                              n_frames, duration_ms, [c[1] for c in cams]))
        det_total = sum(len(c) for c in per_cam)
        print(f"  [{'OK ' if ok else 'MISS'}] {name:20s} expect {expect:6s} -> {decision['verdict']:6s}"
              f" | tracks(2cam)={det_total} fused={decision['fused_counts']}"
              f" unseen={decision['unseen_paid_items']} {duration_ms:.0f}ms")
        for f in decision["flags"]:
            print(f"           FLAG: {f['detail']}")
        for r in decision["review_items"]:
            print(f"           REVIEW: {r['detail']}")
    n_ok = sum(r[3] for r in results)
    print(f"\n  {n_ok}/{len(results)} scenarios as expected")

    if out_path:
        Path(out_path).write_text(json.dumps(observations, ensure_ascii=False, indent=2))
        print(f"  wrote {len(observations)} observation(s) -> {out_path}")
    return results, observations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/images")
    ap.add_argument("--cutouts", default="out/cut_rembg")
    ap.add_argument("--weights", default="runs/detector/best.pt")
    ap.add_argument("--onnx", default="dino_arc.onnx")  # DINOv2+ArcFace (padded, enriched gallery)
    ap.add_argument("--device", default="0")
    ap.add_argument("--frames", type=int, default=4, help="frames per camera")
    ap.add_argument("--out", default=None, help="write VisionObservation JSON here")
    args = ap.parse_args()
    dev = 0 if args.device.isdigit() else args.device
    run(args.dataset, args.cutouts, args.weights, args.onnx, dev,
        out_path=args.out, n_frames=args.frames)


if __name__ == "__main__":
    main()
