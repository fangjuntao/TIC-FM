"""Init file for adapters.

Keep this lightweight for inference-only usage.
"""

from .var_selector import VarianceBasedSelector

__all__ = ["VarianceBasedSelector"]
