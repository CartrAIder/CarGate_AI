"""VISION SIDE end-to-end demo: capture -> detect -> embed -> per-camera tracks
-> cross-camera fusion -> VisionObservation (docs/CONTRACT_v1.1.md §3).

Boundary rule (contract §1): this script answers "what is there, how many".
It does NOT decide whether the cart matches the receipt — no verdict, no band,
no similarity threshold. The demo self-test at the bottom calls the DECISION
side's reference implementation (cartgate.verification.reference_verify) purely
to check the vision output is good enough to decide on; production wiring hands
the JSON to the teammate's VerificationService instead.

Fusion strategy is chosen by calibration: gate_calib.json present ->
PlaneMatchFusion (instances are real physical objects, cross_camera_resolved
true), absent -> AsymmetricFusion (per-camera detections, the decision layer
falls back to its conservative mode).
"""
import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cartgate import config, vision_fusion
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_frames
from cartgate.match import sku_similarity
from cartgate.verification import reference_verify   # demo self-test only

CALIB_PATH = "gate_calib.json"
EMBED_BATCH = 8          # measured sweet spot on L40S: 1.72 ms/crop (vs 3.6 looped)

# How a track's per-frame similarities collapse into ONE number per SKU.
#   "max"  — best available view. Semantically right under occlusion, but biased
#            by track length: max over 5 frames beats max over 2 for the same object.
#   "mean" — length-unbiased, but frames where the item is buried dilute the one
#            clear view that identifies it.
#   "top2" — mean of the two best frames: keeps the clear views, damps the
#            single-lucky-frame outlier that "max" rewards.
# This interacts with the decision layer's SIM_STRONG/SIM_WEAK, so the two must
# be calibrated together on real footage (contract §1: thresholds are decision-owned).
CAND_AGG = "top2"


def load_cutouts(cut_dir: str) -> dict:
    cut = defaultdict(list)
    for p in sorted(Path(cut_dir).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def load_fusion(calib_path: str = CALIB_PATH):
    """PlaneMatchFusion when calibrated, AsymmetricFusion otherwise.

    gate_calib.json format (written by the calibration tool, still to come):
        {"homographies": {"cam_left": [[..3x3..]], "cam_right": [[..]]},
         "merge_radius_cm": 12.0}
    """
    p = Path(calib_path)
    if not p.exists():
        return vision_fusion.AsymmetricFusion()
    calib = json.loads(p.read_text())
    H = {c: np.array(m, np.float64) for c, m in calib["homographies"].items()}
    return vision_fusion.PlaneMatchFusion(
        H, merge_radius_cm=float(calib.get("merge_radius_cm",
                                           vision_fusion.MERGE_RADIUS_CM)))


def _aggregate(sims: list[float], how: str | None = None) -> float:
    """Collapse one track's per-frame similarities for a single SKU. See CAND_AGG."""
    how = how or CAND_AGG
    if not sims:
        return 0.0
    if how == "max":
        return float(max(sims))
    if how == "mean":
        return float(np.mean(sims))
    return float(np.mean(sorted(sims, reverse=True)[:2]))     # top2


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_camera(model, frames, embedder, gallery, receipt_skus, dev,
                   camera_id: str = "cam0") -> list[vision_fusion.Detection]:
    """One camera's frames -> one Detection per tracked object.

    Crops are embedded in batches (the ONNX graph has a dynamic batch axis),
    and every receipt SKU's similarity is kept — the decision layer's global
    assignment needs the full vector, not the argmax. Candidates are restricted
    to this cart's receipt by design (contract §3); never the full catalog.

    Object identity across frames uses IoU association to the synthetic GT as a
    stand-in for ByteTrack (which would track from real 30fps video).
    """
    rc = sorted(set(str(s) for s in receipt_skus))
    prefix = "L" if camera_id.endswith("left") else ("R" if camera_id.endswith("right") else "T")
    tracks = defaultdict(lambda: {"cand": defaultdict(list), "n": 0,
                                  "boxes": [], "confs": []})
    for f in frames:
        res = model.predict(f.image, conf=config.DET_CONF, verbose=False, device=dev)[0]
        hits = []                                    # (track_id, box, det_conf, crop)
        for box, det_conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            bx = tuple(int(v) for v in box)
            oid, best = None, config.TRACK_IOU       # associate detection -> GT object id
            for o in f.objects:
                j = _iou(bx, o.box)
                if j > best:
                    best, oid = j, o.track_id
            if oid is None:
                continue                             # spurious detection (no GT) -> skip
            crop = f.image[max(0, bx[1]):bx[3], max(0, bx[0]):bx[2]]
            if crop.size:
                hits.append((oid, bx, float(det_conf), crop))

        for s in range(0, len(hits), EMBED_BATCH):   # batched embedding
            chunk = hits[s:s + EMBED_BATCH]
            vecs = embedder.embed_batch([h[3] for h in chunk])
            for (oid, bx, det_conf, _), vec in zip(chunk, vecs):
                t = tracks[oid]
                t["n"] += 1
                for sku in rc:
                    t["cand"][sku].append(sku_similarity(vec, gallery, sku))
                t["boxes"].append([int(v) for v in bx])
                t["confs"].append(det_conf)

    dets = []
    for tid, t in sorted(tracks.items(), key=lambda kv: str(kv[0])):
        if not t["n"]:
            continue
        dets.append(vision_fusion.Detection(
            camera_id=camera_id,
            track_id=f"{prefix}{tid}",
            candidates={s: round(_aggregate(v), 4) for s, v in sorted(t["cand"].items())},
            n_frames=int(t["n"]),
            box=tuple(t["boxes"][-1]),               # most recent box (closest to the gate)
            det_conf=round(float(np.mean(t["confs"])), 3),
            crop_ref=None,                           # evidence store not wired yet
        ))
    return dets


def run(dataset, cutouts_dir, weights, onnx, dev, seed=7, out_path=None, n_frames=4,
        calib_path=CALIB_PATH):
    rng = np.random.default_rng(seed)
    embedder = get_embedder(onnx, pad=("dino" in onnx or "vit" in onnx))
    print(f"[1/4] embedder: {embedder.name}  (pad={embedder.pad}, "
          f"providers={getattr(embedder, 'providers', ['classical'])}, batch={EMBED_BATCH})")
    print("[2/4] building gallery (studio + synthetic enriched views)...")
    gallery = build_gallery(dataset, embedder, "out", remove_bg=False, enrich_synth=16)
    cutouts = load_cutouts(cutouts_dir)
    model = YOLO(weights)
    fusion = load_fusion(calib_path)
    print(f"[3/4] detector: {weights}  (device={dev})")
    print(f"      fusion: {fusion.name} (cross_camera_resolved={fusion.cross_camera_resolved})")

    skus = sorted(cutouts.keys())
    base = [str(s) for s in rng.choice(skus, size=3, replace=False)]
    extra = next(s for s in skus if s not in base)
    scenarios = [
        ("정상 결제 카트", list(base), {s: 1 for s in base}, "PASS"),
        ("미결제 물건 포함", base + [extra], {s: 1 for s in base}, "FLAG"),
        ("수량 초과(1결제 2적재)", base + [base[0]], {s: 1 for s in base}, "FLAG"),
        ("결제했지만 가려짐", base[:2], {s: 1 for s in base}, "PASS"),
    ]
    cams = [(-1.0, "cam_left"), (1.0, "cam_right")]      # 2 upper-diagonal cameras
    print("[4/4] gate scenarios (real detection -> fusion -> VisionObservation)\n")
    results, observations = [], []
    for i, (name, cart, receipt, expect) in enumerate(scenarios):
        # frames stand in for camera capture -> generated OUTSIDE the timer, so
        # duration_ms measures only detect+recognize+fuse (what a gate would spend).
        cam_frames = [(cam_id, synth_cart_frames(cart, cutouts, rng, n_frames=n_frames,
                                                 size=(640, 640), cam_dirs=[d] * n_frames))
                      for d, cam_id in cams]
        t0 = time.perf_counter()
        per_cam = {cam_id: resolve_camera(model, frames, embedder, gallery,
                                          list(receipt.keys()), dev, camera_id=cam_id)
                   for cam_id, frames in cam_frames}
        obs = vision_fusion.build_observation(
            per_cam, fusion,
            transaction_id=f"TX-DEMO-{i:03d}",
            gate_id=config.GATE_ID,
            captured_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - t0) * 1000),
            frames_used={c: n_frames for c in per_cam})
        observations.append(obs)

        # --- demo self-test ONLY: the decision side is the teammate's to own ---
        verdict = reference_verify.verify(obs, receipt)
        ok = verdict["verdict"] == expect
        results.append((name, expect, verdict, ok))
        n_inst = len(obs["instances"])
        print(f"  [{'OK ' if ok else 'MISS'}] {name:20s} expect {expect:6s} -> {verdict['verdict']:6s}"
              f" | instances={n_inst} counts={verdict['observed_counts']}"
              f" mode={verdict['decision_mode']} {obs['duration_ms']}ms")
        for r in verdict["reasons"]:
            print(f"           {r['severity']}: {r['code']} {r.get('sku_id') or r.get('instance_id', '')}")
    n_ok = sum(r[3] for r in results)
    print(f"\n  {n_ok}/{len(results)} scenarios as expected (decision = reference impl, demo only)")

    if out_path:
        Path(out_path).write_text(json.dumps(observations, ensure_ascii=False, indent=2))
        print(f"  wrote {len(observations)} VisionObservation(s) -> {out_path}")
    return results, observations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/images")
    ap.add_argument("--cutouts", default="out/cut_rembg")
    ap.add_argument("--weights", default="runs/detector/best.pt")
    ap.add_argument("--onnx", default="dino_arc.onnx")  # DINOv2+ArcFace (padded, enriched gallery)
    ap.add_argument("--device", default="0")
    ap.add_argument("--frames", type=int, default=4, help="frames per camera")
    ap.add_argument("--calib", default=CALIB_PATH, help="gate_calib.json (absent -> asymmetric)")
    ap.add_argument("--out", default=None, help="write VisionObservation JSON here")
    args = ap.parse_args()
    dev = 0 if args.device.isdigit() else args.device
    run(args.dataset, args.cutouts, args.weights, args.onnx, dev,
        out_path=args.out, n_frames=args.frames, calib_path=args.calib)


if __name__ == "__main__":
    main()
