from dataclasses import dataclass


@dataclass(frozen=True)
class RecognitionCandidate:
    sku: str
    similarity: float


@dataclass(frozen=True)
class RecognitionResult:
    best_sku: str | None
    best_similarity: float | None
    candidates: tuple[RecognitionCandidate, ...] = ()

