"""Receipt-independent product recognition over the complete gallery."""

from .models import RecognitionCandidate, RecognitionResult
from .recognizer import CatalogRecognizer

__all__ = ["CatalogRecognizer", "RecognitionCandidate", "RecognitionResult"]
