"""End-to-end demo: detect items in each camera's frames, recognize them against
the cart's receipt, fuse the 3 cameras, and return a PASS/FLAG/REVIEW verdict.

Counting identical duplicates across cameras really needs camera geometry;
max-over-cameras is a calibratable stand-in until that lands.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_frames
from cartgate.match import sku_similarity

# similarity bands (recalibrate on real gate footage)
PASS_SIM, REVIEW_SIM, MIN_FRAMES = 0.55, 0.42, 2


def load_cutouts(cut_dir: str) -> dict:
    cut = defaultdict(list)
    for p in sorted(Path(cut_dir).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def _band(sim):
    return "strong" if sim >= PASS_SIM else ("weak" if sim >= REVIEW_SIM else "none")


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_camera(model, frames, embedder, gallery, receipt_skus, dev) -> list[dict]:
    """Detect over one camera's frames and resolve each object to a receipt SKU.

    Object identity across frames uses IoU association to the synthetic GT as a
    stand-in for ByteTrack (which would track from real 30fps video).
    """
    rc = list(set(receipt_skus))
    tracks = defaultdict(lambda: {"votes": Counter(), "sims": defaultdict(list), "n": 0})
    for f in frames:
        res = model.predict(f.image, conf=0.25, verbose=False, device=dev)[0]
        for box in res.boxes.xyxy.cpu().numpy():
            bx = tuple(int(v) for v in box)
            oid, best = None, 0.4                    # associate detection -> GT object id
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
            sim, sku = max(((sku_similarity(vec, gallery, s), s) for s in rc), default=(0.0, None))
            t = tracks[oid]
            t["n"] += 1
            t["votes"][sku] += 1
            t["sims"][sku].append(sim)
    resolved = []
    for t in tracks.values():
        if not t["votes"]:
            continue
        sku = t["votes"].most_common(1)[0][0]
        sim = float(np.mean(t["sims"][sku]))
        b = _band(sim)
        resolved.append({"sku": sku if b != "none" else None, "sim": round(sim, 3),
                         "band": b, "stable": t["n"] >= MIN_FRAMES})
    return resolved


def fuse_and_decide(per_cam: list[list[dict]], receipt: dict[str, int]) -> dict:
    """Fuse the 3 cameras (max-count per SKU) and return the 3-way verdict."""
    def maxcount(pred):
        return max((sum(1 for r in cam if pred(r)) for cam in per_cam), default=0)

    seen_skus = set(receipt) | {r["sku"] for cam in per_cam for r in cam if r["sku"]}
    fused = {s: maxcount(lambda r, s=s: r["sku"] == s and r["band"] == "strong" and r["stable"])
             for s in seen_skus}
    unrec = maxcount(lambda r: r["band"] == "none" and r["stable"])   # matches nothing paid
    weak = maxcount(lambda r: r["band"] == "weak" and r["stable"])    # ambiguous

    flags, reviews = [], []
    for s, paid in receipt.items():
        if fused.get(s, 0) > paid:
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
            "unseen_paid_items": unseen}


def run(dataset, cutouts_dir, weights, onnx, dev, seed=7):
    rng = np.random.default_rng(seed)
    embedder = get_embedder(onnx, pad=("dino" in onnx or "vit" in onnx))
    print(f"[1/4] embedder: {embedder.name}  (pad={embedder.pad})")
    print("[2/4] building gallery (studio + synthetic enriched views)...")
    gallery = build_gallery(dataset, embedder, "out", remove_bg=False, enrich_synth=16)
    cutouts = load_cutouts(cutouts_dir)
    model = YOLO(weights)
    print(f"[3/4] detector: {weights}  (device={dev})")

    skus = sorted(cutouts.keys())
    base = list(rng.choice(skus, size=3, replace=False))
    extra = next(s for s in skus if s not in base)
    scenarios = [
        ("정상 결제 카트", list(base), {s: 1 for s in base}, "PASS"),
        ("미결제 물건 포함", base + [extra], {s: 1 for s in base}, "FLAG"),
        ("수량 초과(1결제 2적재)", base + [base[0]], {s: 1 for s in base}, "FLAG"),
        ("결제했지만 가려짐", base[:2], {s: 1 for s in base}, "PASS"),
    ]
    print("[4/4] gate scenarios (real detection -> 3-camera fusion -> decision)\n")
    results = []
    for name, cart, receipt, expect in scenarios:
        per_cam = []
        for _ in range(3):                       # 1 top + 2 side cameras
            frames = synth_cart_frames(cart, cutouts, rng, n_frames=4, size=(640, 640))
            per_cam.append(resolve_camera(model, frames, embedder, gallery,
                                          list(receipt.keys()), dev))
        d = fuse_and_decide(per_cam, receipt)
        ok = d["verdict"] == expect
        results.append((name, expect, d, ok))
        det_total = sum(len(c) for c in per_cam)
        print(f"  [{'OK ' if ok else 'MISS'}] {name:20s} expect {expect:6s} -> {d['verdict']:6s}"
              f" | tracks(3cam)={det_total} fused={d['fused_counts']}"
              f" unseen={d['unseen_paid_items']}")
        for f in d["flags"]:
            print(f"           FLAG: {f['detail']}")
        for r in d["review_items"]:
            print(f"           REVIEW: {r['detail']}")
    n_ok = sum(r[3] for r in results)
    print(f"\n  {n_ok}/{len(results)} scenarios as expected")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/images")
    ap.add_argument("--cutouts", default="out/cut_rembg")
    ap.add_argument("--weights", default="runs/detect/runs/prod_det_gpu/weights/best.pt")
    ap.add_argument("--onnx", default="dino_arc.onnx")  # DINOv2+ArcFace (padded, enriched gallery)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    dev = 0 if args.device.isdigit() else args.device
    run(args.dataset, args.cutouts, args.weights, args.onnx, dev)


if __name__ == "__main__":
    main()
