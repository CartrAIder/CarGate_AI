"""Build the SKU embedding gallery from dataset/<sku>/<images>. Each image is
expanded into rotation/flip variants so one studio shot still matches field crops
seen at arbitrary rotation."""
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from cartgate.segment import remove_background, crop_to_object

ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]


def rotate_rgba(rgba: np.ndarray, deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    side = int(np.ceil(np.hypot(h, w)))
    canvas = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgba
    M = cv2.getRotationMatrix2D((side / 2, side / 2), deg, 1.0)
    return cv2.warpAffine(canvas, M, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def build_gallery(dataset_dir: str, embedder, out_dir: str = "out",
                  save_cutouts: bool = True, remove_bg: bool = False,
                  enrich_synth: int = 0, cutouts_dir: str = "out/cut_rembg",
                  enrich_seed: int = 7) -> dict:
    """Returns {sku: {"vectors": [N,D], "views": [...]}}.

    remove_bg=False (default) embeds the full studio image with 90-deg rotations +
    flip (detected crops are embedded raw the same way); remove_bg=True uses cutouts.
    enrich_synth>0 also appends that many synthetic composite views per cutout
    (object on varied backgrounds), which makes the gallery cover the field-crop
    distribution rather than only clean studio shots.
    """
    dataset = Path(dataset_dir)
    out = Path(out_dir)
    if save_cutouts and remove_bg:
        (out / "cutouts").mkdir(parents=True, exist_ok=True)

    syn_cut = {}
    if enrich_synth > 0:
        import cv2 as _cv2
        for p in sorted(Path(cutouts_dir).glob("*.png")):
            im = _cv2.imread(str(p), _cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                syn_cut.setdefault(p.stem.split("__")[0], []).append(im)

    gallery: dict = {}

    for sku_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        # key by sku_id (folder prefix before "__"), as receipts/products.csv do
        sku = sku_dir.name.split("__")[0]
        vecs, views = [], []
        for img_path in sorted(sku_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            if remove_bg:
                rgba, _ = remove_background(bgr)
                rgba = crop_to_object(rgba)
                if save_cutouts:
                    cv2.imwrite(str(out / "cutouts" / f"{sku}__{img_path.stem}.png"), rgba)
                for deg in ROTATIONS:
                    rot = rotate_rgba(rgba, deg) if deg else rgba
                    vecs.append(embedder.embed(rot[:, :, :3], rot[:, :, 3]))
                    views.append({"src": img_path.name, "rot": deg, "flip": False})
                flipped = cv2.flip(rgba, 1)
                vecs.append(embedder.embed(flipped[:, :, :3], flipped[:, :, 3]))
                views.append({"src": img_path.name, "rot": 0, "flip": True})
            else:
                # no background removal: embed the whole image + clean 90-deg rotations
                for deg, rot in ((0, bgr), (90, cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)),
                                 (180, cv2.rotate(bgr, cv2.ROTATE_180)),
                                 (270, cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))):
                    vecs.append(embedder.embed(rot, None))
                    views.append({"src": img_path.name, "rot": deg, "flip": False})
                vecs.append(embedder.embed(cv2.flip(bgr, 1), None))
                views.append({"src": img_path.name, "rot": 0, "flip": True})
        if enrich_synth > 0 and sku in syn_cut:
            from cartgate.train_embed import composite
            rng = np.random.default_rng(enrich_seed + hash(sku) % 10000)
            for co in syn_cut[sku]:
                for _ in range(enrich_synth):
                    comp = composite(co, rng)                     # 224 bgr, object on varied bg + degrade
                    vecs.append(embedder.embed(comp, None))
                    views.append({"src": f"synth:{sku}", "rot": 0, "flip": False})

        if vecs:
            gallery[sku] = {"vectors": np.stack(vecs), "views": views}
            print(f"  {sku}: {sum(1 for v in views if str(v['src']).startswith('synth'))} synth + "
                  f"{sum(1 for v in views if not str(v['src']).startswith('synth'))} studio "
                  f"-> {len(vecs)} gallery vectors")

    with open(out / "gallery.pkl", "wb") as f:
        pickle.dump(gallery, f)
    with open(out / "gallery_meta.json", "w") as f:
        json.dump({k: {"n_vectors": int(v["vectors"].shape[0])} for k, v in gallery.items()}, f, indent=2)
    return gallery
