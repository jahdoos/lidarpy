"""Bin point cloud data onto 2D angular or 3D voxel grids at regular time intervals.

Timestamps in column 5 are expected in microseconds (µs).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GridResult:
    """Result of gridding point cloud data.

    Attributes:
        grids: (T, H, W) array of reflectivity values; NaN = no data.
        timestamps: (T,) array of grid timestamps in microseconds (center of each interval).
        az_edges: (W+1,) azimuth bin edges in degrees.
        el_edges: (H+1,) elevation bin edges in degrees.
    """
    grids: np.ndarray       # (T, H, W) float64
    timestamps: np.ndarray  # (T,) float64
    az_edges: np.ndarray    # (W+1,) float64
    el_edges: np.ndarray    # (H+1,) float64


def points_to_grids(
    points: np.ndarray,
    interval_ms: float = 100.0,
    az_range: tuple[float, float] = (-60.0, 60.0),
    el_range: tuple[float, float] = (-12.5, 12.5),
    az_res: float = 0.5,
    el_res: float = 0.5,
) -> GridResult:
    """Convert Nx6 point array to time-sliced reflectivity grids.

    Args:
        points: Nx6 array [x, y, z, reflectivity, tag, timestamp_us].
        interval_ms: Time bin width in milliseconds.
        az_range: (min, max) azimuth in degrees. HAP default ~±60°.
        el_range: (min, max) elevation in degrees. HAP default ~±12.5°.
        az_res: Azimuth resolution in degrees per pixel.
        el_res: Elevation resolution in degrees per pixel.

    Returns:
        GridResult with (T, H, W) reflectivity grids and metadata.
    """
    if points.shape[0] == 0 or points.shape[1] < 6:
        w = int(round((az_range[1] - az_range[0]) / az_res))
        h = int(round((el_range[1] - el_range[0]) / el_res))
        return GridResult(
            grids=np.empty((0, h, w), dtype=np.float64),
            timestamps=np.empty(0, dtype=np.float64),
            az_edges=np.linspace(az_range[0], az_range[1], w + 1),
            el_edges=np.linspace(el_range[0], el_range[1], h + 1),
        )

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    refl = points[:, 3]
    ts_us = points[:, 5]

    # Filter zero-distance points (no-return from LiDAR)
    dist = x**2 + y**2 + z**2
    nonzero = dist > 0
    x, y, z, refl, ts_us = x[nonzero], y[nonzero], z[nonzero], refl[nonzero], ts_us[nonzero]

    if len(ts_us) == 0:
        n_az = int(round((az_range[1] - az_range[0]) / az_res))
        n_el = int(round((el_range[1] - el_range[0]) / el_res))
        return GridResult(
            grids=np.full((1, n_el, n_az), np.nan, dtype=np.float64),
            timestamps=np.array([points[:, 5].mean()]),
            az_edges=np.linspace(az_range[0], az_range[1], n_az + 1),
            el_edges=np.linspace(el_range[0], el_range[1], n_el + 1),
        )

    # --- Compute angular coordinates ---
    r_xy = np.sqrt(x**2 + y**2)
    azimuth = np.degrees(np.arctan2(x, y))       # degrees, 0 = forward (+y)
    elevation = np.degrees(np.arctan2(z, r_xy))   # degrees, 0 = horizontal

    # --- Build spatial bin edges ---
    n_az = int(round((az_range[1] - az_range[0]) / az_res))
    n_el = int(round((el_range[1] - el_range[0]) / el_res))
    az_edges = np.linspace(az_range[0], az_range[1], n_az + 1)
    el_edges = np.linspace(el_range[0], el_range[1], n_el + 1)

    # --- Bin assignment: searchsorted(right) puts upper-edge values in last bin ---
    az_idx = np.searchsorted(az_edges, azimuth, side="right") - 1
    el_idx = np.searchsorted(el_edges, elevation, side="right") - 1

    valid = (
        (az_idx >= 0) & (az_idx < n_az) &
        (el_idx >= 0) & (el_idx < n_el) &
        np.isfinite(azimuth) & np.isfinite(elevation)
    )

    az_idx = az_idx[valid]
    el_idx = el_idx[valid]
    refl = refl[valid]
    ts_us = ts_us[valid]

    if len(ts_us) == 0:
        return GridResult(
            grids=np.full((1, n_el, n_az), np.nan, dtype=np.float64),
            timestamps=np.array([points[:, 5].mean()]),
            az_edges=az_edges,
            el_edges=el_edges,
        )

    # --- Build time bins (timestamps in µs, interval in ms → µs) ---
    interval_us = interval_ms * 1e3
    t_min = ts_us.min()
    t_max = ts_us.max()
    n_frames = max(1, int(np.ceil((t_max - t_min) / interval_us)))
    t_idx = np.clip(
        ((ts_us - t_min) / interval_us).astype(np.int64),
        0, n_frames - 1
    )

    # --- Accumulate into grids (mean reflectivity per cell) ---
    grids_sum = np.zeros((n_frames, n_el, n_az), dtype=np.float64)
    grids_cnt = np.zeros((n_frames, n_el, n_az), dtype=np.float64)

    np.add.at(grids_sum, (t_idx, el_idx, az_idx), refl)
    np.add.at(grids_cnt, (t_idx, el_idx, az_idx), 1.0)

    with np.errstate(invalid="ignore"):
        grids = np.where(grids_cnt > 0, grids_sum / grids_cnt, np.nan)

    # Timestamps at center of each interval (µs)
    timestamps = t_min + (np.arange(n_frames) + 0.5) * interval_us

    return GridResult(
        grids=grids,
        timestamps=timestamps,
        az_edges=az_edges,
        el_edges=el_edges,
    )


@dataclass
class VoxelResult:
    """Result of voxelizing point cloud data.

    Attributes:
        voxels: (T, nx, ny, nz) array of mean reflectivity; NaN = no data.
        timestamps: (T,) array of voxel grid timestamps in microseconds (center of each interval).
        x_edges: (nx+1,) x bin edges in meters.
        y_edges: (ny+1,) y bin edges in meters.
        z_edges: (nz+1,) z bin edges in meters.
    """
    voxels: np.ndarray       # (T, nx, ny, nz) float64
    timestamps: np.ndarray   # (T,) float64
    x_edges: np.ndarray      # (nx+1,) float64
    y_edges: np.ndarray      # (ny+1,) float64
    z_edges: np.ndarray      # (nz+1,) float64


def points_to_voxels(
    points: np.ndarray,
    shape: tuple[int, int, int] = (512, 512, 16),
    interval_ms: float = 100.0,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> VoxelResult:
    """Convert Nx6 point array to time-sliced 3D voxel grids of reflectivity.

    Args:
        points: Nx6 array [x, y, z, reflectivity, tag, timestamp_us].
        shape: (nx, ny, nz) voxel grid dimensions.
        interval_ms: Time bin width in milliseconds.
        x_range: (min, max) x bounds in meters. Auto-fit from data if None.
        y_range: (min, max) y bounds in meters. Auto-fit from data if None.
        z_range: (min, max) z bounds in meters. Auto-fit from data if None.

    Returns:
        VoxelResult with (T, nx, ny, nz) reflectivity voxels and metadata.
    """
    nx, ny, nz = shape

    if points.shape[0] == 0 or points.shape[1] < 6:
        return VoxelResult(
            voxels=np.empty((0, nx, ny, nz), dtype=np.float64),
            timestamps=np.empty(0, dtype=np.float64),
            x_edges=np.zeros(nx + 1),
            y_edges=np.zeros(ny + 1),
            z_edges=np.zeros(nz + 1),
        )

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    refl = points[:, 3]
    ts_us = points[:, 5]

    # Filter zero-distance points (no-return from LiDAR)
    dist = x**2 + y**2 + z**2
    nonzero = dist > 0
    x, y, z, refl, ts_us = x[nonzero], y[nonzero], z[nonzero], refl[nonzero], ts_us[nonzero]

    if len(ts_us) == 0:
        return VoxelResult(
            voxels=np.full((1, nx, ny, nz), np.nan, dtype=np.float64),
            timestamps=np.array([points[:, 5].mean()]),
            x_edges=np.zeros(nx + 1),
            y_edges=np.zeros(ny + 1),
            z_edges=np.zeros(nz + 1),
        )

    # --- Auto-fit spatial ranges with small padding ---
    def _auto_range(vals, rng):
        if rng is not None:
            return rng
        lo, hi = vals.min(), vals.max()
        pad = max((hi - lo) * 0.01, 0.01)
        return (lo - pad, hi + pad)

    x_range = _auto_range(x, x_range)
    y_range = _auto_range(y, y_range)
    z_range = _auto_range(z, z_range)

    x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
    y_edges = np.linspace(y_range[0], y_range[1], ny + 1)
    z_edges = np.linspace(z_range[0], z_range[1], nz + 1)

    # --- Bin assignment: searchsorted(right) puts upper-edge values in last bin ---
    xi = np.searchsorted(x_edges, x, side="right") - 1
    yi = np.searchsorted(y_edges, y, side="right") - 1
    zi = np.searchsorted(z_edges, z, side="right") - 1

    valid = (
        (xi >= 0) & (xi < nx) &
        (yi >= 0) & (yi < ny) &
        (zi >= 0) & (zi < nz)
    )

    xi = xi[valid]
    yi = yi[valid]
    zi = zi[valid]
    refl = refl[valid]
    ts_us = ts_us[valid]

    if len(ts_us) == 0:
        return VoxelResult(
            voxels=np.full((1, nx, ny, nz), np.nan, dtype=np.float64),
            timestamps=np.array([points[:, 5].mean()]),
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
        )

    # --- Build time bins (timestamps in µs, interval in ms → µs) ---
    interval_us = interval_ms * 1e3
    t_min = ts_us.min()
    t_max = ts_us.max()
    n_frames = max(1, int(np.ceil((t_max - t_min) / interval_us)))
    t_idx = np.clip(
        ((ts_us - t_min) / interval_us).astype(np.int64),
        0, n_frames - 1
    )

    # --- Accumulate (mean reflectivity per voxel) ---
    vol_sum = np.zeros((n_frames, nx, ny, nz), dtype=np.float64)
    vol_cnt = np.zeros((n_frames, nx, ny, nz), dtype=np.float64)

    np.add.at(vol_sum, (t_idx, xi, yi, zi), refl)
    np.add.at(vol_cnt, (t_idx, xi, yi, zi), 1.0)

    with np.errstate(invalid="ignore"):
        voxels = np.where(vol_cnt > 0, vol_sum / vol_cnt, np.nan)

    timestamps = t_min + (np.arange(n_frames) + 0.5) * interval_us

    return VoxelResult(
        voxels=voxels,
        timestamps=timestamps,
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
    )
