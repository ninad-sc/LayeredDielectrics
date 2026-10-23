"""Reflection, transmission and absorbed power in layered dielectrics.

Numerical imports do not initialize plotting or create output directories.
"""

from .api import ReflectionResult, calculate_reflection

__version__ = "0.1.0"
__all__ = ["ReflectionResult", "calculate_reflection"]
