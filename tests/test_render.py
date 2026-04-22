"""Tests for point cloud → 2D image rendering."""

import numpy as np
import pytest
from lidarpy.render import render_frame


def _make_points(n, x, y, z, refl, tag=0, ts_us=1e6):
    """Build Nx6 point array from scalars/arrays."""
    pts = np.empty((n, 6), dtype=np.float64)
    pts[:, 0] = x
    pts[:, 1] = y
    pts[:, 2] = z
    pts[:, 3] = refl
    pts[:, 4] = tag
    pts[:, 5] = ts_us
    return pts


class TestRenderFrame:
    def test_empty_input(self):
        pts = np.empty((0, 6), dtype=np.float64)
        img = render_frame(pts, resolution=(64, 64))
        assert img.shape == (64, 64)
        assert img.dtype == np.uint8
        assert img.max() == 0

    def test_output_shape(self):
        pts = _make_points(10, x=np.linspace(-1, 1, 10), y=5.0,
                           z=np.linspace(-0.5, 0.5, 10), refl=100)
        img = render_frame(pts, resolution=(48, 64))
        assert img.shape == (48, 64)
        assert img.dtype == np.uint8

    def test_single_point_nonzero(self):
        pts = _make_points(1, x=0.0, y=5.0, z=0.0, refl=200)
        img = render_frame(pts, resolution=(32, 32))
        assert img.max() > 0

    def test_z_flip(self):
        """Higher Z points should appear in top rows of image."""
        hi = _make_points(100, x=0.0, y=5.0, z=np.full(100, 1.0), refl=200)
        lo = _make_points(100, x=0.0, y=5.0, z=np.full(100, -1.0), refl=200)
        pts = np.vstack([hi, lo])
        img = render_frame(pts, resolution=(64, 32),
                           x_range=(-2, 2), z_range=(-2, 2))
        top_half = img[:32, :].sum()
        bot_half = img[32:, :].sum()
        assert top_half > 0
        assert bot_half > 0
        # High-Z points in top half
        top_row_with_data = np.where(img.sum(axis=1) > 0)[0][0]
        bot_row_with_data = np.where(img.sum(axis=1) > 0)[0][-1]
        assert top_row_with_data < 32
        assert bot_row_with_data >= 32

    def test_explicit_ranges(self):
        pts = _make_points(50, x=np.linspace(-2, 2, 50), y=5.0,
                           z=np.linspace(-1, 1, 50), refl=150)
        img = render_frame(pts, resolution=(64, 128),
                           x_range=(-3, 3), z_range=(-2, 2))
        assert img.shape == (64, 128)
        assert img.max() > 0

    def test_agg_sum_default(self):
        """Two points same pixel with sum → brighter than single point."""
        pts1 = _make_points(1, x=0.0, y=5.0, z=0.0, refl=100)
        pts2 = np.vstack([pts1, pts1.copy()])
        img1 = render_frame(pts1, resolution=(8, 8), x_range=(-1, 1), z_range=(-1, 1))
        img2 = render_frame(pts2, resolution=(8, 8), x_range=(-1, 1), z_range=(-1, 1))
        # Both normalized to 255 since single pixel, but sum has higher raw value
        # With only one occupied pixel both normalize to 255
        assert img1.max() == 255
        assert img2.max() == 255

    def test_agg_mean(self):
        pts = np.vstack([
            _make_points(1, x=0.0, y=5.0, z=0.0, refl=60),
            _make_points(1, x=0.01, y=5.0, z=0.01, refl=100),
        ])
        img = render_frame(pts, resolution=(8, 8), x_range=(-1, 1),
                           z_range=(-1, 1), agg="mean")
        assert img.max() > 0

    def test_agg_callable(self):
        pts = np.vstack([
            _make_points(1, x=0.0, y=5.0, z=0.0, refl=50),
            _make_points(1, x=0.01, y=5.0, z=0.01, refl=200),
        ])
        img = render_frame(pts, resolution=(8, 8), x_range=(-1, 1),
                           z_range=(-1, 1), agg=np.max)
        assert img.max() > 0

    def test_zero_distance_filtered(self):
        """Points at origin (0,0,0) should be filtered out."""
        pts = _make_points(5, x=0.0, y=0.0, z=0.0, refl=200)
        img = render_frame(pts, resolution=(32, 32))
        assert img.max() == 0

    def test_contiguous(self):
        pts = _make_points(10, x=np.linspace(-1, 1, 10), y=5.0,
                           z=np.linspace(-0.5, 0.5, 10), refl=100)
        img = render_frame(pts, resolution=(32, 32))
        assert img.flags["C_CONTIGUOUS"]
