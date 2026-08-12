from collections import Counter
from collections.abc import Iterable

from .models import (Decision, ObservationStatus, PaidItem, UnexpectedItem,
                     VerificationResult, VisionObservation)


def verify_items(paid_items: Iterable[PaidItem], observation: VisionObservation) -> VerificationResult:
    """Compare aggregated SKU quantities; paid-but-unseen items are tolerated."""
    if observation.status is not ObservationStatus.VALID:
        reason = observation.failure_reason or f"vision observation is {observation.status.value.lower()}"
        return VerificationResult(Decision.REVIEW, reasons=(reason,))

    paid, detected = Counter(), Counter()
    for item in paid_items:
        paid[item.sku] += item.quantity
    unknown = tuple(sorted((item for item in observation.products if item.sku is None),
                           key=lambda item: (item.quantity,
                                             item.detection_confidence is None,
                                             item.detection_confidence or 0.0,
                                             item.recognition_similarity is None,
                                             item.recognition_similarity or 0.0)))
    for item in observation.products:
        if item.sku is not None:
            detected[item.sku] += item.quantity

    unexpected = tuple(UnexpectedItem(sku, paid[sku], quantity, quantity - paid[sku])
                       for sku, quantity in sorted(detected.items()) if quantity > paid[sku])
    reasons = [f"unpaid or excess {item.sku}: {item.unexpected_quantity} item(s)" for item in unexpected]
    if unknown:
        reasons.append(f"{sum(item.quantity for item in unknown)} unknown item(s) detected")
    decision = Decision.REVIEW if unexpected or unknown else Decision.PASS
    return VerificationResult(decision, unexpected, unknown, tuple(reasons))
