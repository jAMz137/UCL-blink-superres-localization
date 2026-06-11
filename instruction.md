# Instruction

## Algorithm Overview

The input is a time series of wide-field luminescence image frames. The stack is treated as a 3D matrix with time, y, and x axes. Blinking transitions are identified from frame differences, and each accepted transition is used to isolate one emitter from a dense diffraction-limited spot.

The bundled demo data are cropped from the original wide-field image data and correspond to spot region I in the manuscript. The crop keeps the demo compact, but small numerical differences from manuscript outputs are expected.

The workflow follows these stages:

1. Load each indexed `*stack.tif` file from `data/demo_stacks_36px`.
2. Interpret the file index from 0 to 10 as increasing excitation power.
3. Load the matched fiducial marker stack from `data/driftmark5V`.
4. Estimate drift in 20-frame segments by cross-correlating averaged marker frames.
5. Smooth the image stack with a 3D Gaussian filter and compute frame differences.
6. Detect local maxima in the positive and negative frame-difference stacks.
7. Convert each candidate into a `BlinkEvent` and reject candidates with malformed extent, weak fitting, or boundary overlap.
8. Pair same-polarity events that cancel each other and mark them as glitch regions.
9. For each remaining event, find the nearest preceding and following events in a local spatiotemporal search volume.
10. Use those neighbors to define the two non-blinking intervals immediately before and after the target event.
11. Average stable frames from both intervals and subtract the two averaged spots to isolate the single-emitter signal.
12. Fit the isolated spot with a 2D Gaussian, correct the fitted position by drift, and keep localizations that pass quality filters.
13. Save the localization table, scatter image, Gaussian-rendered density map, and cluster-wise localization precision. Localization coordinates are reported in camera pixels; the configured pixel size is used to convert precision estimates to nanometres.

All demo stacks are processed in one run.

## Module Guide

### `TRIM_analysis.py`

Main entry point. It parses command-line arguments, discovers all demo stack files, matches each stack to its marker file, and calls the event, interval, drift, fitting, and rendering modules.

Input:

- `data/demo_stacks_36px/*stack.tif`
- `data/driftmark5V/*stamark.tif`

Output:

- `results/localizations.txt`
- `results/scatter_ploti.png`
- `results/Gaussianrender_density_map_raw.PNG`
- `results/localization_precision.txt`

### `src/TRIM_config.py`

Defines `AnalysisConfig`. The exposed routine settings are `pixel_size_nm` and `spot_sigma_px`. Fixed thresholds and search windows live in the modules that use them.

### `src/TRIM_events.py`

Defines `BlinkEvent` and `BlinkEventSet`.

`BlinkEvent` stores the spatiotemporal extent, centroid, screening status, and glitch flag for one blinking transition. `BlinkEventSet` detects positive and negative events from the Gaussian-smoothed frame-difference stack and builds the glitch mask used by the interval extractor.

### `src/TRIM_intervals.py`

Defines `NonBlinkingInterval` and `NonBlinkingIntervalSet`.

`NonBlinkingInterval` stores the local time trace, the two non-blinking time ranges, the differential spot, and localization metadata. `NonBlinkingIntervalSet` searches for neighboring events around each target event, extracts local image stacks, selects stable frames, and computes the differential single-particle spot.

### `src/drift_correction.py`

Defines `DriftCorrection`. The marker stack is split into 20-frame segments. Each segment is averaged, background-normalized, cross-correlated with a reference, and fitted with a 2D Gaussian to estimate x/y drift.

### `src/TRIM_render.py`

Saves the final localization products. It writes the localization table, renders the cluster-numbered scatter plot, renders a Gaussian density map with coordinate axes, and estimates localization precision by Gaussian mixture clustering followed by weighted center and uncertainty calculation.

### `src/utils_toolbox.py`

Contains low-level numerical utilities: consecutive-frame masks, monotonic range search, circular masks, local-maxima detection, and 2D Gaussian fitting.

## Running in a Configured Environment

Run `TRIM_analysis.py` from the repository root in an environment configured with the required packages. The default arguments process `data/` and write to `results/`. To change the two routine parameters from Python, call:

```python
from pathlib import Path
from TRIM_analysis import run_pipeline

run_pipeline(
    data_dir=Path("data"),
    output_dir=Path("results"),
    pixel_size_nm=40,
    spot_sigma_px=5,
)
```
