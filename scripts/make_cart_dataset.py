"""Generate the synthetic cart dataset — one build serving two purposes:

  1) decision-layer benchmark:  carts.json (receipt vs contents, per-camera GT, verdict)
  2) detector fine-tuning:       YOLO labels + data.yaml (single "product" class)

Two upper-diagonal cameras (opposite sides), scenarios mix normal / unpaid /
quantity-excess so verdicts are known.

  python scripts/make_cart_dataset.py --num 3000     # 3000 carts x 2 cams = 6000 imgs
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from cartgate.synth import synth_cart_frames


def load_cutouts(d="out/cut_rembg"):
    cut = defaultdict(list)
    for p in sorted(Path(d).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def make_cart(skus, rng):
    """Return (receipt {sku:qty}, contents [sku...], scenario, verdict, reason)."""
    k = int(rng.integers(2, 6))
    paid = list(rng.choice(skus, size=k, replace=False))
    receipt = {s: int(rng.integers(1, 3)) for s in paid}
    contents = [s for s, q in receipt.items() for _ in range(q)]
    r = rng.random()
    if r < 0.60:
        return receipt, contents, "normal", "PASS", "cart matches receipt"
    if r < 0.80:
        extra = next(s for s in rng.permutation(skus) if s not in receipt)
        contents.append(extra)
        return receipt, contents, "unpaid_item", "FLAG", f"unpaid item present: {extra}"
    over = paid[int(rng.integers(len(paid)))]
    contents.append(over)
    return receipt, contents, "quantity_excess", "FLAG", f"more '{over}' than paid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=3000, help="carts (each = `cameras` images)")
    ap.add_argument("--cameras", type=int, default=2)
    ap.add_argument("--out", default="cart_dataset")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--val-frac", type=float, default=0.08)
    args = ap.parse_args()

    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    cut = load_cutouts()
    skus = list(cut.keys())
    cam_dirs = list(np.linspace(-1.0, 1.0, args.cameras)) if args.cameras > 1 else [0.0]

    carts, counts = [], Counter()
    for i in range(args.num):
        rng = np.random.default_rng(1000 + i)
        receipt, contents, scenario, verdict, reason = make_cart(skus, rng)
        counts[scenario] += 1
        frames = synth_cart_frames(contents, cut, rng, n_frames=args.cameras,
                                   size=(args.size, args.size), cam_dirs=cam_dirs)
        split = "val" if rng.random() < args.val_frac else "train"
        cams = []
        for c, f in enumerate(frames):
            name = f"cart_{i:05d}_cam{c}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), f.image)
            lines = []
            for o in f.objects:
                x0, y0, x1, y1 = o.box
                cx, cy = (x0 + x1) / 2 / args.size, (y0 + y1) / 2 / args.size
                bw, bh = (x1 - x0) / args.size, (y1 - y0) / args.size
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            cams.append({"image": f"images/{split}/{name}.jpg",
                         "objects": [{"sku": o.sku, "box": list(o.box)} for o in f.objects]})
        carts.append({"id": f"cart_{i:05d}", "scenario": scenario, "verdict": verdict,
                      "reason": reason, "receipt": receipt, "contents": dict(Counter(contents)),
                      "cameras": cams})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{args.num}", flush=True)

    json.dump(carts, open(out / "carts.json", "w"), ensure_ascii=False, indent=1)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: product\n")
    ntr = len(list((out / "images" / "train").glob("*.jpg")))
    nva = len(list((out / "images" / "val").glob("*.jpg")))
    print(f"\n{args.num} carts x {args.cameras} cams = {args.num * args.cameras} images -> {out}/")
    print("scenarios:", dict(counts), "| train/val imgs:", ntr, nva)


if __name__ == "__main__":
    main()
