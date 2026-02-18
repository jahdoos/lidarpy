"""Tests for point cloud → 2D grid and 3D voxel conversion."""

import numpy as np
import pytest
from lidarpy.grid import points_to_grids, GridResult, points_to_voxels, VoxelResult


def _make_points(n, x, y, z, refl, tag, ts_us):
    """Build Nx6 point array from scalars/arrays."""
    pts = np.empty((n, 6), dtype=np.float64)
    pts[:, 0] = x
    pts[:, 1] = y
    pts[:, 2] = z
    pts[:, 3] = refl
    pts[:, 4] = tag
    pts[:, 5] = ts_us
    return pts


class TestGrid:
    def test_empty_input(self):
        pts = np.empty((0, 6), dtype=np.float64)
        r = points_to_grids(pts)
        assert r.grids.shape[0] == 0
        assert r.timestamps.shape[0] == 0
        assert r.grids.shape[1] > 0  # H
        assert r.grids.shape[2] > 0  # W

    def test_single_point_center(self):
        # Point at (0, 5, 0) → azimuth=0°, elevation=0°, refl=100
        pts = _make_points(1, x=0, y=5.0, z=0, refl=100, tag=0, ts_us=1e6)
        r = points_to_grids(pts, interval_ms=100, az_res=1.0, el_res=1.0)
        assert r.grids.shape[0] == 1  # 1 time frame
        # Center pixel should be ~100
        h, w = r.grids.shape[1], r.grids.shape[2]
        center_val = r.grids[0, h // 2, w // 2]
        assert center_val == 100.0

    def test_two_time_bins(self):
        # Two points 200ms apart → 2 grids at 100ms interval
        pts = np.vstack([
            _make_points(1, 0, 5.0, 0, 80, 0, ts_us=0),
            _make_points(1, 0, 5.0, 0, 120, 0, ts_us=200e3),  # 200ms = 200000 µs
        ])
        r = points_to_grids(pts, interval_ms=100, az_res=1.0, el_res=1.0)
        assert r.grids.shape[0] == 2
        assert len(r.timestamps) == 2
        assert r.timestamps[1] > r.timestamps[0]

    def test_mean_reflectivity(self):
        # Two points same cell, different refl → average
        pts = np.vstack([
            _make_points(1, 0, 5.0, 0, 60, 0, ts_us=1e6),
            _make_points(1, 0, 5.0, 0, 100, 0, ts_us=1e6 + 1e3),
        ])
        r = points_to_grids(pts, interval_ms=100, az_res=1.0, el_res=1.0)
        h, w = r.grids.shape[1], r.grids.shape[2]
        assert r.grids[0, h // 2, w // 2] == pytest.approx(80.0)

    def test_nan_for_empty_cells(self):
        pts = _make_points(1, x=0, y=5.0, z=0, refl=100, tag=0, ts_us=1e6)
        r = points_to_grids(pts, interval_ms=100, az_res=1.0, el_res=1.0)
        # Most cells should be NaN
        total = r.grids[0].size
        nan_count = np.isnan(r.grids[0]).sum()
        assert nan_count > total - 5  # at most a few cells filled

    def test_out_of_fov_ignored(self):
        # Point at azimuth ~80° (outside default ±60° range)
        # x/y = tan(80°) ≈ 5.67 → point at (5.67, 1.0, 0)
        pts = _make_points(1, x=5.67, y=1.0, z=0, refl=200, tag=0, ts_us=1e6)
        r = points_to_grids(pts, interval_ms=100, az_res=1.0, el_res=1.0)
        # Should be all NaN since point is outside FOV
        assert np.all(np.isnan(r.grids[0]))

    def test_result_shapes(self):
        pts = _make_points(100, x=np.random.randn(100), y=5+np.random.randn(100)*0.1,
                           z=np.random.randn(100)*0.1, refl=np.random.rand(100)*255,
                           tag=0, ts_us=np.linspace(0, 500e3, 100))
        r = points_to_grids(pts, interval_ms=100, az_res=0.5, el_res=0.5,
                            az_range=(-60, 60), el_range=(-12.5, 12.5))
        assert r.grids.ndim == 3
        assert r.grids.shape[1] == 50   # 25° / 0.5°
        assert r.grids.shape[2] == 240  # 120° / 0.5°
        assert len(r.az_edges) == 241
        assert len(r.el_edges) == 51

    def test_timestamps_centered(self):
        pts = _make_points(10, x=0, y=5.0, z=0, refl=100, tag=0,
                           ts_us=np.linspace(0, 50e3, 10))
        r = points_to_grids(pts, interval_ms=100)
        # Single bin, timestamp at center of interval
        assert r.timestamps[0] == pytest.approx(50e3, rel=0.5)


class TestVoxel:
    def test_empty_input(self):
        pts = np.empty((0, 6), dtype=np.float64)
        r = points_to_voxels(pts, shape=(8, 8, 4))
        assert r.voxels.shape == (0, 8, 8, 4)
        assert len(r.timestamps) == 0

    def test_output_shape(self):
        pts = _make_points(50,
                           x=np.random.uniform(-5, 5, 50),
                           y=np.random.uniform(0, 10, 50),
                           z=np.random.uniform(-1, 1, 50),
                           refl=np.random.rand(50) * 255,
                           tag=0,
                           ts_us=np.linspace(0, 50e3, 50))
        r = points_to_voxels(pts, shape=(16, 16, 8), interval_ms=100)
        assert r.voxels.shape == (1, 16, 16, 8)
        assert len(r.timestamps) == 1
        assert len(r.x_edges) == 17
        assert len(r.y_edges) == 17
        assert len(r.z_edges) == 9

    def test_single_point(self):
        # Point at (1.0, 2.0, 0.5), refl=150
        pts = _make_points(1, x=1.0, y=2.0, z=0.5, refl=150, tag=0, ts_us=1e6)
        r = points_to_voxels(pts, shape=(4, 4, 4), interval_ms=100,
                             x_range=(0, 4), y_range=(0, 4), z_range=(0, 4))
        assert r.voxels.shape == (1, 4, 4, 4)
        # Exactly one voxel should be 150, rest NaN
        filled = ~np.isnan(r.voxels[0])
        assert filled.sum() == 1
        assert r.voxels[0][filled][0] == 150.0

    def test_mean_reflectivity(self):
        # Two points same voxel, refl 60 and 100 → mean 80
        pts = np.vstack([
            _make_points(1, x=1.0, y=2.0, z=0.5, refl=60, tag=0, ts_us=1e6),
            _make_points(1, x=1.05, y=2.05, z=0.55, refl=100, tag=0, ts_us=1e6 + 1e3),
        ])
        r = points_to_voxels(pts, shape=(4, 4, 4), interval_ms=100,
                             x_range=(0, 4), y_range=(0, 4), z_range=(0, 4))
        filled = ~np.isnan(r.voxels[0])
        assert filled.sum() == 1
        assert r.voxels[0][filled][0] == pytest.approx(80.0)

    def test_two_time_bins(self):
        pts = np.vstack([
            _make_points(1, x=1.0, y=2.0, z=0.5, refl=100, tag=0, ts_us=0),
            _make_points(1, x=1.0, y=2.0, z=0.5, refl=200, tag=0, ts_us=200e3),
        ])
        r = points_to_voxels(pts, shape=(4, 4, 4), interval_ms=100,
                             x_range=(0, 4), y_range=(0, 4), z_range=(0, 4))
        assert r.voxels.shape[0] == 2
        assert len(r.timestamps) == 2
        assert r.timestamps[1] > r.timestamps[0]

    def test_auto_range(self):
        pts = _make_points(100,
                           x=np.random.uniform(-3, 3, 100),
                           y=np.random.uniform(1, 8, 100),
                           z=np.random.uniform(-0.5, 0.5, 100),
                           refl=128, tag=0,
                           ts_us=np.linspace(0, 50e3, 100))
        r = points_to_voxels(pts, shape=(32, 32, 8))
        # Edges should span the data
        assert r.x_edges[0] <= pts[:, 0].min()
        assert r.x_edges[-1] >= pts[:, 0].max()
        assert r.y_edges[0] <= pts[:, 1].min()
        assert r.y_edges[-1] >= pts[:, 1].max()

    def test_out_of_range_ignored(self):
        pts = _make_points(1, x=10.0, y=10.0, z=10.0, refl=200, tag=0, ts_us=1e6)
        r = points_to_voxels(pts, shape=(4, 4, 4), interval_ms=100,
                             x_range=(0, 4), y_range=(0, 4), z_range=(0, 4))
        assert np.all(np.isnan(r.voxels[0]))

    def test_nan_for_empty_voxels(self):
        pts = _make_points(1, x=1.0, y=2.0, z=0.5, refl=100, tag=0, ts_us=1e6)
        r = points_to_voxels(pts, shape=(8, 8, 4), interval_ms=100,
                             x_range=(0, 8), y_range=(0, 8), z_range=(0, 4))
        total = r.voxels[0].size
        nan_count = np.isnan(r.voxels[0]).sum()
        assert nan_count >= total - 1

    def test_large_shape(self):
        # Verify 512x512x16 shape works (don't fill much data)
        pts = _make_points(10,
                           x=np.random.uniform(-5, 5, 10),
                           y=np.random.uniform(0, 10, 10),
                           z=np.random.uniform(-1, 1, 10),
                           refl=100, tag=0,
                           ts_us=np.linspace(0, 50e3, 10))
        r = points_to_voxels(pts, shape=(512, 512, 16), interval_ms=100)
        assert r.voxels.shape == (1, 512, 512, 16)
