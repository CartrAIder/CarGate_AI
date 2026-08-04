"""Synthesize cart scenes from single-object cutouts for detector training.
Objects are piled with overlap, perspective/rotation/lighting jitter, motion blur
and quality degradation. GT boxes come from the composition (occlusion-aware: an
object mostly covered by items on top of it is dropped), so no manual boxing."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from cartgate.gallery import rotate_rgba


@dataclass
class PlacedObject:
    sku: str
    track_id: int
    box: tuple  # x0, y0, x1, y1


@dataclass
class Frame:
    image: np.ndarray
    objects: list = field(default_factory=list)


def random_degrade(img: np.ndarray, rng: np.random.Generator, p: float = 0.65) -> np.ndarray:
    """Degrade like a cheap/far/low-light camera (low-res, noise, defocus,
    brightness, JPEG). Geometry is preserved so GT boxes stay valid."""
    if rng.random() > p:
        return img
    h, w = img.shape[:2]
    if rng.random() < 0.6:                                   # low resolution
        s = rng.uniform(0.35, 0.7)
        img = cv2.resize(cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s)))), (w, h))
    if rng.random() < 0.4:                                   # extra defocus
        img = cv2.GaussianBlur(img, (int(rng.choice([3, 5, 7])),) * 2, 0)
    if rng.random() < 0.5:                                   # sensor noise
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(5, 20), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.45:                                  # brightness / low light
        img = np.clip(img.astype(np.float32) * rng.uniform(0.5, 1.2), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                                   # JPEG blocking
        _, e = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 60))])
        img = cv2.imdecode(e, 1)
    return img


def make_cart_background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full((h, w, 3), rng.integers(90, 150), np.uint8)
    noise = rng.normal(0, 12, (h, w, 1)).astype(np.float32)
    bg = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    step = rng.integers(40, 60)                       # cart wire mesh
    color = int(rng.integers(60, 90))
    for x in range(0, w, step):
        cv2.line(bg, (x, 0), (x, h), (color,) * 3, 2)
    for y in range(0, h, step):
        cv2.line(bg, (0, y), (w, y), (color,) * 3, 2)
    return cv2.GaussianBlur(bg, (5, 5), 0)


def _motion_blur(img: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Directional motion blur -> the cart is rolling past the camera."""
    if rng.random() >= p:
        return img
    L = int(rng.integers(7, 21))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2, L / 2), float(rng.uniform(0, 180)), 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    s = k.sum()
    return cv2.filter2D(img, -1, k / s) if s > 0 else img


def _perspective_jitter(rgba: np.ndarray, rng: np.random.Generator, amt: float = 0.13) -> np.ndarray:
    """Random perspective warp -> same product seen from top vs side vantage."""
    h, w = rgba.shape[:2]
    d = amt * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgba, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _paste_alpha(canvas: np.ndarray, rgba: np.ndarray, cx: int, cy: int):
    """Alpha-composite rgba centred at (cx,cy); return this object's alpha over the
    FULL canvas (for occlusion bookkeeping), or None if it landed off-frame."""
    h, w = rgba.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    H, W = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (alpha * patch[:, :, :3] + (1 - alpha) * region).astype(np.uint8)
    full = np.zeros(canvas.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = patch[:, :, 3]
    return full


def synth_cart_frames(cart_contents: list[str], cutouts: dict[str, list[np.ndarray]],
                      rng: np.random.Generator, n_frames: int = 3,
                      size: tuple = (960, 720), vis_thresh: float = 0.18) -> list[Frame]:
    """cart_contents: list of SKU ids (repeats = quantity).
    cutouts: sku -> list of RGBA cutouts (one per available view).
    vis_thresh: drop an object's box if less than this fraction stays visible.
    """
    W, H = size
    frames = []
    for _ in range(n_frames):
        canvas = make_cart_background(W, H, rng)
        placed = []  # (track_id, sku, full-canvas alpha)
        ccx, ccy = rng.uniform(0.35, 0.65) * W, rng.uniform(0.35, 0.65) * H  # pile centre
        for track_id in rng.permutation(len(cart_contents)):
            sku = cart_contents[track_id]
            rgba = cutouts[sku][rng.integers(len(cutouts[sku]))].copy()
            target = rng.uniform(0.14, 0.42) * min(W, H)          # near/far scale
            s = target / max(rgba.shape[:2])
            rgba = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if rng.random() < 0.6:
                rgba = _perspective_jitter(rgba, rng)             # multi-view
            rgba = rotate_rgba(rgba, float(rng.uniform(0, 360)))
            f = rng.uniform(0.7, 1.25)                            # lighting
            col = rng.uniform(0.9, 1.1, 3)                        # colour temp
            rgba[:, :, :3] = np.clip(rgba[:, :, :3].astype(np.float32) * f * col, 0, 255).astype(np.uint8)
            if rng.random() < 0.75:                               # clustered pile (overlap)
                cx = int(np.clip(rng.normal(ccx, 0.17 * W), 0, W))
                cy = int(np.clip(rng.normal(ccy, 0.17 * H), 0, H))
            else:                                                 # near edge (partial view)
                cx, cy = int(rng.uniform(0.05, 0.95) * W), int(rng.uniform(0.05, 0.95) * H)
            alpha = _paste_alpha(canvas, rgba, cx, cy)
            if alpha is not None and alpha.any():
                placed.append((int(track_id), sku, alpha))

        # occlusion-aware boxes: an item is hidden by everything dropped AFTER it
        frame = Frame(image=canvas)
        for idx, (tid, sku, alpha) in enumerate(placed):
            occ = np.zeros((H, W), bool)
            for _, _, a2 in placed[idx + 1:]:
                occ |= a2 > 0
            vis = (alpha > 0) & ~occ
            area = int((alpha > 0).sum())
            if area == 0 or vis.sum() / area < vis_thresh:        # mostly buried -> no label
                continue
            ys, xs = np.where(vis)
            frame.objects.append(PlacedObject(
                sku=sku, track_id=tid,
                box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))

        frame.image = _motion_blur(frame.image, rng)              # rolling cart
        k = 3 if rng.random() < 0.5 else 5
        frame.image = cv2.GaussianBlur(frame.image, (k, k), 0)
        frames.append(frame)
    return frames
