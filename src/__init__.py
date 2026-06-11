"""TRIM package for blinking-event super-resolution localization.

The root-level ``TRIM_analysis.py`` script is the public entry point. Modules
inside this package provide event detection, non-blinking interval extraction,
drift correction, Gaussian fitting helpers, and rendering utilities.
"""

from .TRIM_config import AnalysisConfig

__all__ = ["AnalysisConfig"]
__version__ = "0.1.0"
