"""Single source of truth for the runtime thresholds.

These were previously duplicated in scripts/pipeline.py, scripts/viz_recognition.py
and cartgate/match.py with three different value sets. Import them from here so a
deployment recalibration is a one-file change.

All of these are calibrated on synthetic carts and MUST be re-tuned on real gate
footage before deployment.
"""

# --- recognition similarity bands (cosine similarity against the receipt gallery) ---
PASS_SIM = 0.55      # >= this: confident enough to count as a paid item
REVIEW_SIM = 0.42    # >= this but < PASS_SIM: ambiguous -> human review

# --- track stability ---
MIN_FRAMES = 2       # a track seen in fewer frames never accuses anyone

# --- detector ---
DET_CONF = 0.25      # YOLO confidence threshold
TRACK_IOU = 0.4      # IoU needed to associate a detection with a tracked object

# --- gate identity (overridden per deployment) ---
GATE_ID = "demo-gate-1"


def band(sim: float) -> str:
    """strong / weak / none, the confidence band a similarity falls into."""
    return "strong" if sim >= PASS_SIM else ("weak" if sim >= REVIEW_SIM else "none")
