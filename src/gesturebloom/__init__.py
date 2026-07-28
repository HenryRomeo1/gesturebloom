"""GestureBloom: gesture-driven procedural spiderlily.

Real-time hand landmarks drive a parametric flower on the GPU, with a causal
temporal model for gesture-event detection.

Only numpy-dependent symbols are exported at package level so that
``import gesturebloom`` never triggers a torch, mediapipe, or moderngl import.
"""

from .landmarks.canonical import HandFrame, canonicalize, feature_vector, try_canonicalize
from .landmarks.filters import LandmarkSmoother, OneEuroFilter

__version__ = "0.1.0"

__all__ = [
    "HandFrame",
    "LandmarkSmoother",
    "OneEuroFilter",
    "__version__",
    "canonicalize",
    "feature_vector",
    "try_canonicalize",
]
