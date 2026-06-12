# UCL Blinking-Based Super-Resolution Localization

This repository implements a super-resolution localization method for blinking upconversion luminescence. The algorithm detects turn-on and turn-off blinking events in dense luminescence image stacks, extracts the stable non-blinking intervals before and after each event, isolates the single-particle differential spot, corrects sample drift, fits a 2D Gaussian, and renders the final localization point cloud.

Although developed for upconversion luminescence nanocrystals, the same workflow should be extendable to the broader class of blinking-event-based super-resolution localization problems. 

Because the method was designed from the perspective of processing three-dimensional blinking time traces, the algorithm is being planned to be named **TRIM (Time tRace Integrated Microscopy).** The acronym TRIM also captures the key step of trimming non-blinking intervals from three-dimensional data.

The demo stacks are indexed from `0stack.tif` to `10stack.tif`, with matched drift-marker files named `0stamark.tif` to `10stamark.tif`. Index 0 is the lowest excitation power and index 10 is the highest; the approximate power-density range is 2 to 4 kW/cm^2. The demo data are cropped from the original wide-field image data and correspond to spot region I in the manuscript. Because the released demo uses this cropped region, the numerical output can differ slightly from the manuscript values.

## System Requirements

- Operating system: Windows 10/11 is the tested target environment.
- Python: tested with version `3.13.9`.
- Required packages: `numpy==2.3.4`, `scipy==1.16.3`, `matplotlib==3.10.7`, `tifffile==2025.10.16`, and `scikit-learn==1.7.2`.
- Hardware: no non-standard hardware is required. The demo data are image stacks, so enough RAM for several hundred MB of TIFF data is recommended.

## Installation Guide

Create and activate an environment, then install the pinned dependencies from `requirements.txt`:

```powershell
conda create -n trim-env python=3.13.9
conda activate trim-env
conda install --file requirements.txt -c conda-forge
```

From the repository root, verify the installation:

```powershell
python -m py_compile TRIM_analysis.py src/TRIM_config.py src/TRIM_events.py src/TRIM_intervals.py src/TRIM_render.py src/drift_correction.py src/utils_toolbox.py
```

Typical installation time on a normal desktop computer is a few minutes, depending on Conda package cache status and network speed.

## Demo and Usage

Run the complete demo once from the repository root:

```powershell
python TRIM_analysis.py --data-dir data --output-dir results --pixel-size-nm 40 --spot-sigma-px 5
```

The only routine user-facing parameters are:

- `--pixel-size-nm`: camera pixel size in nanometres. Default: `40`.
- `--spot-sigma-px`: expected diffraction-limited spot sigma in pixels. Default: `5`.

All other thresholds are built into the modules to keep the published demo reproducible.

Expected outputs in `results/`:

- `localizations.txt`: x/y localization table in pixel coordinates with fitting weights and mean fitting error.
- `scatter_ploti.png`: scatter rendering of accepted localization points with numbered clusters.
- `Gaussianrender_density_map_raw.PNG`: Gaussian-rendered density map with coordinate axes.
- `localization_precision.txt`: cluster-wise weighted localization precision using the same numbering as the scatter plot.

Expected runtime depends on CPU and disk speed. On a normal desktop computer, the full bundled demo may take several minutes because all TIFF stacks are processed in one run.

## Repository Layout

- `TRIM_analysis.py`: main program and dataset-level orchestration.
- `src/TRIM_config.py`: user-facing pixel-size and spot-sigma settings.
- `src/TRIM_events.py`: blinking-event detection and glitch masking.
- `src/TRIM_intervals.py`: non-blinking interval extraction and differential spot generation.
- `src/drift_correction.py`: fiducial-marker drift correction.
- `src/TRIM_render.py`: localization table export, scatter rendering, Gaussian rendering, and precision estimation.
- `src/utils_toolbox.py`: Gaussian fitting and array utilities.
- `instruction.md`: detailed algorithm and module guide.

## License

This project is released under the GNU General Public License v3.0. See `LICENSE`.
