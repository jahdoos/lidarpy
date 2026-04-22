"""Render point cloud frames as 2D grayscale images (front-facing XZ projection)."""

from collections.abc import Callable

import numpy as np


def render_frame(
    points: np.ndarray,
    resolution: tuple[int, int] = (480, 640),
    x_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    agg: str | Callable = "sum",
) -> np.ndarray:
    """Project Nx6 point cloud onto HxW uint8 image via front-facing XZ projection.

    X → column, Z → row (flipped so +Z = top). Pixel intensity from reflectivity.

    Args:
        points: Nx6 array [x, y, z, reflectivity, tag, timestamp_us].
        resolution: (H, W) output image size.
        x_range: (min, max) x bounds in meters. Auto-fit if None.
        z_range: (min, max) z bounds in meters. Auto-fit if None.
        agg: Aggregation per pixel — "sum" (default), "mean", or callable(array)->scalar.

    Returns:
        HxW uint8 grayscale image. Empty pixels = 0.
    """
    h, w = resolution

    if points.shape[0] == 0 or points.shape[1] < 4:
        return np.zeros((h, w), dtype=np.uint8)

    x, z, refl = points[:, 0], points[:, 2], points[:, 3]

    # Filter zero-distance points (no-return)
    dist_sq = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
    nonzero = dist_sq > 0
    x, z, refl = x[nonzero], z[nonzero], refl[nonzero]

    if len(x) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Auto-fit ranges
    def _fit(vals, rng):
        if rng is not None:
            return rng
        lo, hi = vals.min(), vals.max()
        pad = max((hi - lo) * 0.01, 0.01)
        return (lo - pad, hi + pad)

    x_range = _fit(x, x_range)
    z_range = _fit(z, z_range)

    x_edges = np.linspace(x_range[0], x_range[1], w + 1)
    z_edges = np.linspace(z_range[0], z_range[1], h + 1)

    ci = np.searchsorted(x_edges, x, side="right") - 1
    ri = np.searchsorted(z_edges, z, side="right") - 1

    valid = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    ci, ri, refl = ci[valid], ri[valid], refl[valid]

    if len(ci) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Aggregate
    if callable(agg) and not isinstance(agg, str):
        grid = np.full((h, w), np.nan, dtype=np.float64)
        flat = ri * w + ci
        order = np.argsort(flat)
        flat_s, refl_s = flat[order], refl[order]
        splits = np.searchsorted(flat_s, np.arange(h * w), side="left")
        splits = np.append(splits, len(flat_s))
        for idx in range(h * w):
            if splits[idx] < splits[idx + 1]:
                grid.flat[idx] = agg(refl_s[splits[idx]:splits[idx + 1]])
    elif agg == "mean":
        grid_sum = np.zeros((h, w), dtype=np.float64)
        grid_cnt = np.zeros((h, w), dtype=np.float64)
        np.add.at(grid_sum, (ri, ci), refl)
        np.add.at(grid_cnt, (ri, ci), 1.0)
        with np.errstate(invalid="ignore"):
            grid = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan)
    else:  # "sum" default
        grid_sum = np.zeros((h, w), dtype=np.float64)
        grid_cnt = np.zeros((h, w), dtype=np.float64)
        np.add.at(grid_sum, (ri, ci), refl)
        np.add.at(grid_cnt, (ri, ci), 1.0)
        grid = np.where(grid_cnt > 0, grid_sum, np.nan)

    # Normalize to 0-255
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    mx = finite.max()
    if mx <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    scaled = np.where(np.isfinite(grid), grid / mx * 255.0, 0.0)
    img = np.clip(scaled, 0, 255).astype(np.uint8)

    # Flip so +Z = top of image
    img = img[::-1]
    return np.ascontiguousarray(img)
