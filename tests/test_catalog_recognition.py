import numpy as np

from cartgate.catalog_recognition import CatalogRecognizer


def test_recognizes_across_entire_catalog_and_returns_top_k():
    gallery = {
        "water": {"vectors": np.array([[1.0, 0.0]], dtype=np.float32)},
        "coke": {"vectors": np.array([[0.8, 0.6]], dtype=np.float32)},
        "snack": {"vectors": np.array([[0.0, 1.0]], dtype=np.float32)},
    }
    result = CatalogRecognizer(gallery, top_k=2).recognize(np.array([0.9, 0.1]))
    assert result.best_sku == "water"
    assert [candidate.sku for candidate in result.candidates] == ["water", "coke"]


def test_low_similarity_is_unknown_but_candidates_are_preserved():
    gallery = {"water": {"vectors": np.array([[1.0, 0.0]], dtype=np.float32)}}
    result = CatalogRecognizer(gallery, min_similarity=0.5).recognize(np.array([0.0, 1.0]))
    assert result.best_sku is None
    assert result.candidates[0].sku == "water"
