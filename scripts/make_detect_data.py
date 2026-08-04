"""Generate a synthetic 1-class ("product") YOLO dataset from cutouts.

The detector only localizes product-shaped blobs; SKU identity comes later from the
embedding + receipt match. Carts are composited from cutouts, so GT boxes are free
(no manual bounding boxes). Output is the standard Ultralytics YOLO layout.

  python3 make_detect_data.py --cutouts out/cutouts --out detect_data --size 640
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cartgate.synth import synth_cart_frames, random_degrade  # noqa: E402


def load_cutouts(cut_dir: Path) -> dict:
    cut = {}
    for p in sorted(cut_dir.glob("*.png")):
        sku = p.stem.split("__")[0]
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            cut.setdefault(sku, []).append(img)
    return cut


def gen_split(split: str, n: int, cutouts: dict, rng, out: Path, size):
    W, H = size
    imdir = out / "images" / split
    lbdir = out / "labels" / split
    imdir.mkdir(parents=True, exist_ok=True)
    lbdir.mkdir(parents=True, exist_ok=True)
    skus = list(cutouts.keys())
    n_box = 0
    for i in range(n):
        k = int(rng.integers(3, 11))                      # 3-10 items per cart (heavy occlusion)
        contents = list(rng.choice(skus, size=k, replace=True))
        frame = synth_cart_frames(contents, cutouts, rng, n_frames=1, size=size)[0]
        name = f"{split}_{i:05d}"
        img = random_degrade(frame.image, rng)      # low-quality-camera robustness
        cv2.imwrite(str(imdir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        lines = []
        for o in frame.objects:
            x0, y0, x1, y1 = o.box
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            if bw <= 0.002 or bh <= 0.002:
                continue
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(bw, 1.0), min(bh, 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1
        (lbdir / f"{name}.txt").write_text("\n".join(lines))
    print(f"  {split}: {n} images, {n_box} boxes -> {imdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutouts", default="out/cutouts")
    ap.add_argument("--out", default="detect_data")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cutouts = load_cutouts(Path(args.cutouts))
    if not cutouts:
        raise SystemExit(f"no RGBA cutouts in {args.cutouts} (run run_demo.py or seg_compare first)")
    print(f"loaded cutouts for {len(cutouts)} SKUs ({sum(len(v) for v in cutouts.values())} views)")

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    gen_split("train", args.n_train, cutouts, rng, out, (args.size, args.size))
    gen_split("val", args.n_val, cutouts, rng, out, (args.size, args.size))

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames:\n  0: product\n")
    print(f"\n  data.yaml -> {out/'data.yaml'}")
    print(f"  train: yolo detect train model=yolo11n.pt data={out/'data.yaml'} imgsz={args.size} epochs=50")


if __name__ == "__main__":
    main()
