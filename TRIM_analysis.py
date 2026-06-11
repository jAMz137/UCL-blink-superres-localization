"""Command-line entry point for blinking-event super-resolution localization."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import tifffile as tifile
from scipy.ndimage import gaussian_filter

from src.TRIM_config import AnalysisConfig
from src.TRIM_events import BlinkEventSet
from src.TRIM_render import (
    estimate_localization_precision,
    save_density_map,
    save_localization_table,
    save_scatter_plot,
)
from src.TRIM_intervals import NonBlinkingIntervalSet
from src.drift_correction import DriftCorrection
from src.utils_toolbox import sci_opt_fit


def power_index_from_name(path: Path) -> int:
    """Extract the demo power index from names such as 0stack.tif."""

    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot parse power index from {path.name}")
    value = int(numbers[0])
    return value


def marker_path_for(marker_dir: Path, power_index: int) -> Path | None:
    candidate = marker_dir / f"{power_index}stamark.tif"
    if candidate.exists():
        return candidate
    return None


def locate_intervals(
    stack_path: Path,
    marker_path: Path,
    output_dir: Path,
    config: AnalysisConfig,
    power_index: int,
) -> list[dict]:
    """Run event detection, interval extraction, fitting, and drift correction."""

    excitation = power_index
    raw_stack = np.array(np.float32(tifile.imread(stack_path)))[:-1]
    image_stack = raw_stack

    drift = DriftCorrection(
        marker_path.parent,
        marker_path.name,
        power_index,
        20,
        1,
        config.pixel_size,
    ).correlated()

    gaussian_stack = gaussian_filter(
        image_stack,
        sigma=(0.5, config.spot_sigma_px, config.spot_sigma_px),
    )
    image_diff = np.diff(image_stack, axis=0)
    gaussian_diff = np.diff(gaussian_stack, axis=0)
    events = BlinkEventSet(image_diff, gaussian_diff, config)
    intervals = NonBlinkingIntervalSet(events, image_stack, config, excitation)

    localizations = []
    for interval in intervals.Traces:
        if interval.abort == 1:
            continue
        image_to_fit = interval.SpotB if interval.SpotB is not None else interval.SpotA
        if image_to_fit is None:
            continue
        min_y, min_x = interval.corner
        fit = sci_opt_fit(image_to_fit, config.pixel_size, 4.6)
        if fit[1] == "fail":
            interval.abort = 4
            binary_image = (image_to_fit > np.max(image_to_fit) / 3).astype(int)
            coords = np.column_stack(np.nonzero(binary_image))
            if len(coords) == 0:
                continue
            centroid = coords.mean(axis=0)
            raw_x = float(centroid[1])
            raw_y = float(centroid[0])
            continue
        popt, _, perr = fit
        raw_x = float(popt[1])
        raw_y = float(popt[2])
        drift_index = min(int(interval.SpotI // 20), len(drift) - 1)
        drift_x, drift_y = drift[drift_index]
        x = float(raw_x + min_x - drift_x)
        y = float(raw_y + min_y - drift_y)
        ellip = abs(popt[3] / popt[4] - 1)
        fit_error = float(perr[1] + perr[2])

        interval.xy = [x, y]
        interval.drift = [float(drift_x), float(drift_y)]
        interval.fit_o = {
            "stdxy": [float(perr[1]), float(perr[2])],
            "sigma_xy": [float(popt[3]), float(popt[4])],
        }

        if (
            interval.ValidC >= 100
            and ellip < 0.1
            and fit_error < 0.2
            and perr[1] < 0.1
            and perr[2] < 0.1
        ):
            localizations.append(
                {
                    "source": stack_path.name,
                    "excitation": excitation,
                    "frame": float(interval.cnt_z),
                    "x": x,
                    "y": y,
                    "stdx": float(perr[1]),
                    "stdy": float(perr[2]),
                    "valid_counts": float(interval.ValidC),
                }
            )
        else:
            interval.abort = 5

    print(
        f"{stack_path.name}: {len(localizations)} accepted localizations "
        f"from {len(intervals.Traces)} intervals"
    )
    return localizations


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    pixel_size_nm: float,
    spot_sigma_px: float,
) -> None:
    """Process every demo stack once and write all final outputs."""

    config = AnalysisConfig(pixel_size_nm=pixel_size_nm, spot_sigma_px=spot_sigma_px)
    stack_dir = data_dir / "demo_stacks_36px"
    marker_dir = data_dir / "driftmark5V"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_localizations = []
    stack_paths = sorted(stack_dir.glob("*stack.tif"), key=power_index_from_name)
    if not stack_paths:
        raise RuntimeError(f"No demo stacks found in {stack_dir}.")
    image_shape_px = tuple(int(value) for value in tifile.imread(stack_paths[0], key=0).shape)
    for stack_path in stack_paths:
        power_index = power_index_from_name(stack_path)
        marker_path = marker_path_for(marker_dir, power_index)
        if marker_path is None:
            print(f"Skipping {stack_path.name}: missing marker for power index {power_index}")
            continue
        all_localizations.extend(
            locate_intervals(stack_path, marker_path, output_dir, config, power_index)
        )

    if not all_localizations:
        raise RuntimeError("No accepted localizations were produced.")

    table = save_localization_table(all_localizations, output_dir / "localizations.txt")
    cluster_rows = estimate_localization_precision(
        table,
        output_dir / "localization_precision.txt",
        pixel_size_nm=config.pixel_size_nm,
    )
    save_scatter_plot(
        table,
        output_dir / "scatter_ploti.png",
        cluster_rows=cluster_rows,
        image_shape_px=image_shape_px,
    )
    save_density_map(
        table,
        output_dir / "Gaussianrender_density_map_raw.PNG",
        image_shape_px=image_shape_px,
    )
    print(f"Saved {len(table)} localizations to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blinking-event super-resolution localization on all demo stacks."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--pixel-size-nm", type=float, default=40.0)
    parser.add_argument("--spot-sigma-px", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pixel_size_nm=args.pixel_size_nm,
        spot_sigma_px=args.spot_sigma_px,
    )
    print(f"Total runtime: {time.time() - start:.2f} s")
