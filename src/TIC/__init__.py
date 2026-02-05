"""Top-level package exports for `TIC`.

This module intentionally avoids importing submodules that in turn import
`TIC` at module import time (for example `sklearn.classifier`). Importing
those submodules here causes circular imports when users do ``from TIC
import ...``. To prevent that, we export lightweight symbols directly and
provide lazy accessors for heavier sklearn-based classes.
"""

# Export core model classes and config eagerly -- these are pure model modules
# and do not import `TIC` back.
from .model.inference_config import InferenceConfig


__all__ = [
	"InferenceConfig",
	"TSEncoderICLClassifierV2",
]


def __getattr__(name: str):
	"""Lazy-import heavier sklearn wrapper classes on attribute access.

	This avoids circular import errors when importing the package. Example:
		from TIC import TSEncoderICLClassifierV2

	will import the sklearn wrapper only when the attribute is referenced.
	"""
	if name == "TSEncoderICLClassifierV2":
		from .sklearn.classifier import TSEncoderICLClassifierV2

		return TSEncoderICLClassifierV2
	raise AttributeError(f"module 'TIC' has no attribute '{name}'")


def __dir__():
    return __all__