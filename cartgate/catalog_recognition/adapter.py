"""Adapters from existing detector/tracker crops to catalog recognition output."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

from cartgate.verification.models import DetectedProduct, ObservationStatus, VisionObservation

from .recognizer import CatalogRecognizer


@dataclass(frozen=True)
class TrackedCrop:
    track_id: Hashable
    crop: np.ndarray
    detection_confidence: float | None = None


def recognize_tracks(tracks: Iterable[TrackedCrop], embedder, recognizer: CatalogRecognizer,
                     *, min_track_observations: int = 1,
                     cross_camera_duplicates_resolved: bool = True) -> VisionObservation:
    """Vote repeated crops into one physical item per track, then count per SKU.

    Call this after per-camera tracking. If tracks from multiple cameras are mixed
    without cross-camera association, set cross_camera_duplicates_resolved=False;
    verification will fail safe to REVIEW instead of trusting an inflated count.
    """
    votes, similarities, confidences, seen = defaultdict(Counter), defaultdict(list), defaultdict(list), Counter()
    try:
        for tracked in tracks:
            if tracked.crop.size == 0:
                continue
            result = recognizer.recognize(embedder.embed(tracked.crop, None))
            seen[tracked.track_id] += 1
            votes[tracked.track_id][result.best_sku] += 1
            similarities[(tracked.track_id, result.best_sku)].append(result.best_similarity)
            confidences[tracked.track_id].append(tracked.detection_confidence)
    except Exception as exc:
        return VisionObservation.invalid(f"vision adapter failure: {exc}")

    if not cross_camera_duplicates_resolved:
        return VisionObservation.invalid("cross-camera duplication unresolved")

    physical_items = []
    for track_id in sorted(seen, key=str):
        if seen[track_id] < min_track_observations:
            return VisionObservation.invalid(f"unstable track: {track_id}")
        sku = votes[track_id].most_common(1)[0][0]
        sims = [value for value in similarities[(track_id, sku)] if value is not None]
        confs = [value for value in confidences[track_id] if value is not None]
        physical_items.append((sku, np.mean(confs) if confs else None, np.mean(sims) if sims else None))

    grouped = {}
    for sku, det_conf, rec_sim in physical_items:
        bucket = grouped.setdefault(sku, {"quantity": 0, "det": [], "sim": []})
        bucket["quantity"] += 1
        if det_conf is not None:
            bucket["det"].append(float(det_conf))
        if rec_sim is not None:
            bucket["sim"].append(float(rec_sim))
    products = tuple(DetectedProduct(sku, values["quantity"],
                                     min(values["det"]) if values["det"] else None,
                                     min(values["sim"]) if values["sim"] else None)
                     for sku, values in sorted(grouped.items(), key=lambda item: str(item[0])))
    return VisionObservation(products, ObservationStatus.VALID)

