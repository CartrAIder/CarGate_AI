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

Payment API JSON can be converted at the boundary with the strict adapter:

```python
from cartgate.verification import parse_payment_json

payment = parse_payment_json(response.text)
# payment.order_id: str (payment.cart_id is the same compatibility identifier)
# payment.status: "PAID"
# payment.items: tuple[PaidItem, ...]
```

The expected contract is `{"orderId": str, "status": "PAID", "items":
[{"productId": positive int, "productName": str, "quantity": positive int}]}`.
`orderId` is used as the verification cart identifier. Each QuickPass `productId`
is converted to the CartGate SKU with `f"S{product_id:04d}"` (`1 -> S0001`,
`101 -> S0101`). The order-item `id` and price/customer fields are ignored.
`productName` is validated but is not used for matching. An empty `items` array
is a valid empty receipt. Malformed JSON, a non-PAID status, or invalid fields
raise `PaymentJsonError` and must not be treated as an empty payment.

Run the deterministic tests with `python -m pytest -q` from the repository root.
Model weights and camera data are not needed for these unit tests.
