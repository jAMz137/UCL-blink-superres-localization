"""User-facing configuration for the TRIM localization pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Only routine user-facing parameters are kept here."""

    pixel_size_nm: float = 40.0
    spot_sigma_px: float = 5.0

    @property
    def pixel_size(self) -> float:
        """Pixel-unit scale used by the original Gaussian fitting workflow."""

        return 1.0
