from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"


class ObservationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class PaidItem:
    sku: str
    quantity: int

    def __post_init__(self):
        if not self.sku or self.quantity < 0:
            raise ValueError("paid item requires a SKU and non-negative quantity")


@dataclass(frozen=True)
class DetectedProduct:
    sku: str | None
    quantity: int
    detection_confidence: float | None = None
    recognition_similarity: float | None = None

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError("detected product quantity must be positive")


@dataclass(frozen=True)
class VisionObservation:
    products: tuple[DetectedProduct, ...] = ()
    status: ObservationStatus = ObservationStatus.VALID
    failure_reason: str | None = None

    @classmethod
    def invalid(cls, reason: str, *, unavailable: bool = False):
        status = ObservationStatus.UNAVAILABLE if unavailable else ObservationStatus.INVALID
        return cls(status=status, failure_reason=reason)


@dataclass(frozen=True)
class UnexpectedItem:
    sku: str
    paid_quantity: int
    detected_quantity: int
    unexpected_quantity: int


@dataclass(frozen=True)
class VerificationResult:
    decision: Decision
    unexpected_items: tuple[UnexpectedItem, ...] = ()
    unknown_items: tuple[DetectedProduct, ...] = ()
    reasons: tuple[str, ...] = ()
