# Catalog-wide verification flow

This production-oriented path is separate from the existing receipt-conditioned
demo. The detector, embedder, gallery builder, and per-camera tracker remain the
vision team's code; only their tracked crops cross the adapter boundary.

```text
camera -> existing detector -> existing per-camera tracker -> tracked crops
       -> existing embedder -> CatalogRecognizer (all gallery SKUs)
       -> one vote per physical track -> SKU quantity list
       -> PaymentRepository -> VerificationService -> PASS / REVIEW
```

## Usage

```python
from cartgate.catalog_recognition import CatalogRecognizer
from cartgate.catalog_recognition.adapter import TrackedCrop, recognize_tracks
from cartgate.verification import (
    InMemoryPaymentRepository, PaidItem, VerificationService,
)

recognizer = CatalogRecognizer(gallery, min_similarity=0.45, top_k=3)
observation = recognize_tracks(
    tracked_crops,
    embedder,
    recognizer,
    min_track_observations=2,
    cross_camera_duplicates_resolved=True,
)
payments = InMemoryPaymentRepository({"cart-1": [PaidItem("S0001", 1)]})
result = VerificationService(payments).verify("cart-1", observation)
```

`tracked_crops` must contain repeated observations with a stable `track_id`.
Counting is by distinct track, never by frame detection. Pass one camera at a
time unless track IDs have been associated across cameras. When cross-camera
association is unresolved, pass `cross_camera_duplicates_resolved=False`; the
adapter emits an invalid observation and verification returns `REVIEW`.

An empty, valid observation means the vision system explicitly confirmed an
empty cart. Camera, tracking, or observation failures must instead be represented
with `VisionObservation.invalid(...)`, and therefore fail safe to `REVIEW`.

To connect a real payment DB/API, implement only:

```python
class PaymentRepository:
    def get_paid_items(self, cart_id: str) -> list[PaidItem]: ...
```

Run the deterministic tests with `python -m pytest -q` from the repository root.
Model weights and camera data are not needed for these unit tests.
