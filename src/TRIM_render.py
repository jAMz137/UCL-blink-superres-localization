"""Output helpers for localization tables and reconstructed images."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

def save_localization_table(localizations: list[dict], path: Path) -> np.ndarray:
    """Save x, y, fitting weights, and mean fitting error as a text table."""

    rows = []
    for item in localizations:
        stdx = max(float(item["stdx"]), 1e-12)
        stdy = max(float(item["stdy"]), 1e-12)
        rows.append(
            [
                float(item["x"]),
                float(item["y"]),
                1.0 / stdx**2,
                1.0 / stdy**2,
                (stdx + stdy) / 2.0,
            ]
        )
    data = np.asarray(rows, dtype=float)
    header = "x_px y_px weight_x weight_y mean_fit_error_px"
    np.savetxt(path, data, header=header)
    return data


def _cluster_rows(
    data: np.ndarray,
    n_components: int = 8,
    pixel_size_nm: float = 40.0,
) -> list[dict]:
    """Cluster points and return rows ordered from top-to-bottom, left-to-right."""

    labels, is_outlier = _cluster_labels(data, n_components=n_components)

    x = data[:, 0]
    y = data[:, 1]
    wx = data[:, 2]
    wy = data[:, 3]
    rows = []
    for cluster_id in range(int(np.max(labels)) + 1):
        mask = (labels == cluster_id) & (~is_outlier)
        count = int(np.count_nonzero(mask))
        if count < 2:
            continue
        cx = float(np.sum(x[mask] * wx[mask]) / np.sum(wx[mask]))
        cy = float(np.sum(y[mask] * wy[mask]) / np.sum(wy[mask]))
        std_x = float(
            np.sqrt(
                np.sum(wx[mask] * (x[mask] - cx) ** 2)
                / max(np.sum(wx[mask]) * (count - 1), 1e-12)
            )
        )
        std_y = float(
            np.sqrt(
                np.sum(wy[mask] * (y[mask] - cy) ** 2)
                / max(np.sum(wy[mask]) * (count - 1), 1e-12)
            )
        )
        rows.append(
            {
                "raw_cluster_id": cluster_id,
                "count": count,
                "center_x_px": cx,
                "center_y_px": cy,
                "precision_x_nm": std_x * pixel_size_nm,
                "precision_y_nm": std_y * pixel_size_nm,
            }
        )

    rows.sort(key=lambda row: (-row["center_y_px"], row["center_x_px"]))
    for label, row in enumerate(rows, start=1):
        row["cluster_id"] = label
    return rows


def _cluster_labels(data: np.ndarray, n_components: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Assign every localization to a compact spatial cluster."""

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r".*threadpoolctl.*",
        )
        try:
            from sklearn import cluster
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for localization precision estimation."
            ) from exc

        n_components = min(n_components, len(data))
        points = data[:, :2]
        model = cluster.KMeans(
            n_clusters=n_components,
            random_state=42,
            n_init=50,
        )
        labels = model.fit_predict(points)

    return labels, np.zeros(len(data), dtype=bool)


def _cluster_label_map(cluster_rows: list[dict]) -> dict[int, int]:
    """Map raw clustering labels to the published cluster numbers."""

    return {int(row["raw_cluster_id"]): int(row["cluster_id"]) for row in cluster_rows}


def _scatter_label_positions(cluster_rows: list[dict]) -> dict[int, tuple[float, float]]:
    """Place each cluster label at its center plus a fixed right shift."""

    return {
        int(row["cluster_id"]): (
            float(row["center_x_px"]) + 15.0,
            float(row["center_y_px"]),
        )
        for row in cluster_rows
    }


def save_scatter_plot(
    data: np.ndarray,
    path: Path,
    point_radius_px: float = 0.05,
    cluster_rows: list[dict] | None = None,
) -> None:
    """Render the localization point cloud and label the eight clusters."""

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.98)
    if cluster_rows is None:
        cluster_rows = _cluster_rows(data)
    labels, is_outlier = _cluster_labels(data, n_components=len(cluster_rows))
    label_map = _cluster_label_map(cluster_rows)
    colors = plt.get_cmap("tab10")

    for point_index, (x, y) in enumerate(data[:, :2]):
        cluster_number = label_map.get(int(labels[point_index]))
        point_color = "lightgray" if is_outlier[point_index] else colors((cluster_number or 0) - 1)
        ax.add_patch(
            Circle(
                (float(x), float(y)),
                point_radius_px,
                facecolor=point_color,
                edgecolor=point_color,
            )
        )

    label_positions = _scatter_label_positions(cluster_rows)
    for row in cluster_rows:
        label_x, label_y = label_positions.get(
            row["cluster_id"],
            (row["center_x_px"], row["center_y_px"]),
        )
        ax.text(
            label_x,
            label_y,
            str(row["cluster_id"]),
            color="red",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
        )

    ax.set_xlim(0, 36)
    ax.set_ylim(0, 36)
    ax.set_aspect("equal")
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def render_smlm_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    pixel_size_px: float,
    weights: np.ndarray | None = None,
    sigma_px: float = 1.2,
    range_xy: list[list[float]] | None = None,
) -> np.ndarray:
    """Convert localization points into a Gaussian-rendered density map."""

    if range_xy is None:
        range_xy = [[float(x.min()), float(x.max())], [float(y.min()), float(y.max())]]

    nx = max(1, int((range_xy[0][1] - range_xy[0][0]) / pixel_size_px))
    ny = max(1, int((range_xy[1][1] - range_xy[1][0]) / pixel_size_px))
    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=[ny, nx],
        range=range_xy,
        weights=weights,
    )
    return gaussian_filter(hist, sigma=sigma_px)


def save_density_map(data: np.ndarray, path: Path) -> None:
    """Save the normalized Gaussian-rendered density map with coordinate axes."""

    weights = data[:, 2] + data[:, 3]
    weights = weights / weights.max()
    z = render_smlm_gaussian(
        data[:, 0],
        data[:, 1],
        pixel_size_px=75 / 600,
        weights=weights,
        sigma_px=1.2,
        range_xy=[[0, 36], [0, 36]],
    )
    z = np.maximum(z, 1e-12)
    z_log = np.log10(z * 100)
    z_normalized = z_log / z_log.max()

    cmap = copy.copy(plt.get_cmap("gray_r"))
    cmap.set_over("white")
    cmap.set_under("white")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.98)
    ax.imshow(
        np.rot90(z_normalized),
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[0, 36, 0, 36],
        aspect="equal",
        interpolation="none",
    )
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 36)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def estimate_localization_precision(
    data: np.ndarray,
    output_path: Path,
    n_components: int = 8,
    pixel_size_nm: float = 40.0,
) -> list[dict]:
    """Cluster localizations and save weighted precision estimates."""

    rows = _cluster_rows(data, n_components=n_components, pixel_size_nm=pixel_size_nm)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("cluster_id,count,center_x_px,center_y_px,precision_x_nm,precision_y_nm\n")
        for row in rows:
            handle.write(
                "{cluster_id},{count},{center_x_px:.6f},{center_y_px:.6f},"
                "{precision_x_nm:.6f},{precision_y_nm:.6f}\n".format(**row)
            )
    return rows
