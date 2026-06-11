"""Cross-correlation drift correction using fiducial marker stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal as signal
import tifffile as tifile

from .utils_toolbox import sci_opt_fit


class DriftCorrection:
    """Estimate per-segment x/y drift from a stable marker image stack."""

    def __init__(
        self,
        marker_dir: Path,
        marker_file: str,
        power_index: int,
        drift_step: int,
        marker_count: int,
        pixel_size: float,
    ):
        self.marker_dir = Path(marker_dir)
        self.filemark = marker_file
        self.power_index = power_index
        self.pixel_size = pixel_size
        self.drift_step = drift_step
        self.marker_count = marker_count

    def correlated(self) -> np.ndarray:
        """Return average x/y drift for each ``drift_step`` frame segment."""

        drift_all = []
        for _ in range(self.marker_count):
            marker_stack = np.array(
                np.float32(tifile.imread(self.marker_dir / self.filemark))
            )[:, 0:28, 0:28]
            shape = np.shape(marker_stack)
            drift_segments = []
            current_record = self.marker_dir / f"dfting{self.power_index:03d}.txt"
            previous_record = self.marker_dir / f"dfting{self.power_index - 1:03d}.txt"
            current_drift_image = self.marker_dir / f"drift{self.power_index:03d}.tif"
            previous_drift_image = self.marker_dir / f"drift{self.power_index - 1:03d}.tif"
            first_drift_image = self.marker_dir / "drifA000.tif"

            if self.power_index == 0:
                reference = np.mean(marker_stack[: self.drift_step], axis=0)
                reference -= np.min(reference)
                reference = reference / np.max(reference)
                tifile.imwrite(first_drift_image, reference)
            elif previous_drift_image.exists():
                reference = np.array(tifile.imread(previous_drift_image))
            else:
                reference = np.mean(marker_stack[: self.drift_step], axis=0)
                reference -= np.min(reference)
                reference = reference / np.max(reference)

            marker_stack = marker_stack[:-1]
            last_x = 0.0
            last_y = 0.0
            last_label = reference
            for i in range(0, np.shape(marker_stack)[0] // self.drift_step):
                label = np.mean(
                    marker_stack[i * self.drift_step : (i + 1) * self.drift_step],
                    axis=0,
                )
                label -= np.min(label)
                if np.max(label) == 0:
                    drift_segments.append([last_x, last_y])
                    continue
                label = label / np.max(label)
                last_label = label
                corr = signal.convolve(label, reference[::-1, ::-1], mode="same")
                corr = corr[
                    int(shape[1] / 4 * 1) : int(shape[1] / 4 * 3),
                    int(shape[2] / 4 * 1) : int(shape[2] / 4 * 3),
                ]
                fitout = sci_opt_fit(corr, self.pixel_size, 7.5)
                if fitout[1] == "fail":
                    drift_segments.append([last_x, last_y])
                    continue
                last_x = float(fitout[0][1] - corr.shape[0] / 2 * self.pixel_size)
                last_y = float(fitout[0][2] - corr.shape[1] / 2 * self.pixel_size)
                drift_segments.append([last_x, last_y])

            if drift_segments:
                drift_segments.insert(-1, drift_segments[-1])
            drift_array = np.array(drift_segments, dtype=float)
            if self.power_index != 0 and previous_record.exists():
                previous_drift = np.loadtxt(previous_record)
                drift_array += previous_drift[-1]
            np.savetxt(current_record, drift_array)
            tifile.imwrite(current_drift_image, last_label)
            drift_all.append(drift_array)

        return np.array(np.mean(np.array(drift_all), axis=0))


drift_correction = DriftCorrection
