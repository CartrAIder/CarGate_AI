"""Catalog-wide nearest-gallery recognition, independent of a receipt."""
from collections.abc import Mapping

import numpy as np

from .models import RecognitionCandidate, RecognitionResult


class CatalogRecognizer:
    """Compare one embedding with every SKU in an existing CartGate gallery."""

    def __init__(self, gallery: Mapping[str, dict], *, min_similarity: float = 0.0,
                 top_k: int = 3):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._gallery = gallery
        self.min_similarity = min_similarity
        self.top_k = top_k

    def recognize(self, embedding: np.ndarray, *, top_k: int | None = None) -> RecognitionResult:
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be at least 1")
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if query.size == 0 or not np.isfinite(norm) or norm == 0:
            raise ValueError("embedding must be a finite, non-zero vector")
        query = query / norm

        scored = []
        for sku, entry in self._gallery.items():
            vectors = np.asarray(entry.get("vectors", []), dtype=np.float32)
            if vectors.ndim != 2 or not len(vectors) or vectors.shape[1] != query.size:
                continue
            scored.append(RecognitionCandidate(str(sku), float(np.max(vectors @ query))))
        scored.sort(key=lambda candidate: (-candidate.similarity, candidate.sku))
        candidates = tuple(scored[:k])
        if not candidates or candidates[0].similarity < self.min_similarity:
            return RecognitionResult(None, candidates[0].similarity if candidates else None, candidates)
        return RecognitionResult(candidates[0].sku, candidates[0].similarity, candidates)

