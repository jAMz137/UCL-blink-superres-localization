"""Numerical helpers used by the TRIM localization pipeline."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize as opt
from scipy.optimize import OptimizeWarning
from scipy.ndimage import maximum_filter


def consec_T(arr: np.ndarray, n: int) -> np.ndarray:
    """Return True for values that belong to a run of at least ``n`` Trues."""

    length = len(arr)
    result = [False] * length
    for i in range(length):
        if i + n <= length and all(arr[i : i + n]):
            for j in range(i, i + n):
                result[j] = True
    return np.array(result)


def consec_T2(arr: np.ndarray, n: int) -> np.ndarray:
    """Return True at the center of each three-frame True run."""

    length = len(arr)
    result = [False] * length
    for i in range(length - 2):
        if all(arr[i : i + 3]):
            result[i + 1] = True
    return np.array(result)


def consec_T3(arr: np.ndarray, arrr: np.ndarray, n: int) -> np.ndarray:
    """Expand accepted stable frames using a wider supporting mask."""

    length = len(arr)
    result = [False] * length
    for i in range(length - 2):
        if all(arrr[i : i + 3]) and arr[i + 1]:
            result[i + 1] = True
    return np.array(result)


def _Gaussian2D1(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xdata_tuple
    x0 = float(x0)
    y0 = float(y0)
    x = x - x0
    y = y - y0
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    z = amplitude * np.exp(-((x1 / sigma_x) ** 2 + (y1 / sigma_y) ** 2) / 2) + offset
    return z.ravel()


def _Gaussian2D2(xy, a, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    r = a * np.exp(
        -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
    ) + offset
    return r.ravel()


def sci_opt_fit(image: np.ndarray, pixel_size: float, sigma_xy: float):
    """Fit a 2D Gaussian and return parameters, status, and standard errors."""

    c0 = image.shape[0]
    c1 = image.shape[1]
    axis0 = np.linspace(0, c1 - 1, c1) * pixel_size
    axis1 = np.linspace(0, c0 - 1, c0) * pixel_size
    xy = np.meshgrid(axis0, axis1)

    image_max = np.max(image)
    image_min = np.min(image)
    position = np.where(image == image_max)
    pos_x = position[0][0]
    pos_y = position[1][0]
    initial_guess = (
        image_max,
        pos_x * pixel_size,
        pos_y * pixel_size,
        pixel_size * sigma_xy,
        pixel_size * sigma_xy,
        image_min,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = opt.curve_fit(
                _Gaussian2D2,
                xy,
                image.ravel(),
                maxfev=2000,
                p0=initial_guess,
            )
    except Exception:
        return [], "fail", []

    if popt[1] < 0 or popt[1] > c1 or popt[2] < 0 or popt[2] > c0:
        status = "fail"
    else:
        status = "success"
    perr = np.sqrt(np.diag(pcov))
    return popt, status, perr


def imloc_max(
    matrix: np.ndarray,
    tr0: float,
    tr1: float,
    win_size: tuple,
    enl0: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Find local maxima and monotonic event extents in a 3D matrix."""

    if tr0 < 0:
        tr0 = -tr0
        matrix = -matrix

    result_range = []
    result_pos = []
    local_maxima = maximum_filter(matrix, size=win_size)
    max_pos = np.argwhere((matrix == local_maxima) & (matrix > tr0))

    for pos in max_pos:
        range_z = mono_range(matrix[:, pos[1], pos[2]], pos[0], enl0, "both", tr0)
        range_y = mono_range(matrix[pos[0], :, pos[2]], pos[1], enl0, "both", tr0)
        range_x = mono_range(matrix[pos[0], pos[1], :], pos[2], enl0, "both", tr0)

        center_region2d = np.sum(
            matrix[
                range_z[0] : range_z[1] + 1,
                max(0, pos[1] - 1) : pos[1] + 2,
                max(0, pos[2] - 1) : pos[2] + 2,
            ],
            axis=0,
        )
        average_value = np.mean(center_region2d)
        if average_value >= tr1:
            result_range.append(
                (
                    np.array([range_z[0], range_y[0], range_x[0]]),
                    np.array([range_z[1], range_y[1], range_x[1]]),
                )
            )
            result_pos.append(pos)
    return result_range, result_pos


def mono_range(
    array0: np.ndarray,
    pos0: int,
    enlx: int,
    direction: str,
    threshold: float,
) -> tuple[int, int]:
    """Search away from a peak while values decay monotonically above threshold."""

    current_value = array0[pos0]
    if direction in ["both", "positive"]:
        i = 1
        while (
            pos0 + i < array0.shape[0]
            and array0[pos0 + i] > threshold
            and array0[pos0 + i] <= current_value
        ):
            current_value = array0[pos0 + i]
            i += 1
        if i > enlx:
            i = enlx + 1
        range_positive = pos0 + i - 1
    else:
        range_positive = pos0

    current_value = array0[pos0]
    if direction in ["both", "negative"]:
        i = 1
        while (
            pos0 - i >= 0
            and array0[pos0 - i] > threshold
            and array0[pos0 - i] <= current_value
        ):
            current_value = array0[pos0 - i]
            i += 1
        if i > enlx:
            i = enlx + 1
        range_negative = pos0 - i + 1
    else:
        range_negative = pos0

    return range_negative, range_positive


def gen_circle(center: np.ndarray, radius: int, mshape: tuple[int, int]):
    """Return integer coordinates inside a circular mask."""

    cy, cx = center
    x_coords, y_coords = np.meshgrid(
        np.arange(max(int(cx - radius), 0), min(int(cx + radius) + 1, mshape[1])),
        np.arange(max(int(cy - radius), 0), min(int(cy + radius) + 1, mshape[0])),
    )
    distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    circle_mask = distances <= radius
    x_circle = x_coords[circle_mask]
    y_circle = y_coords[circle_mask]
    return x_circle, y_circle
