"""Plot paired webcam image + lidar 2D projection from an acquisition folder.

Usage:
    python examples/plot_paired.py <run_dir> [--index N] [--stride S]
                                   [--save OUT.png] [--view front|pano|bev]

<run_dir> must contain 'images/' (*.jpg) and 'lidar/' (*.npy) with paired
indices like frame_000042.jpg ↔ points_000042.npy.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

_IDX_RE = re.compile(r"(\d+)")


def _filter_nonzero(points: np.ndarray) -> np.ndarray:
    d2 = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
    return points[d2 > 0]


def _scale_to_255(a, mn, mx, gamma=1.0, clip_hi_pct=None):
    """Scale values to 0-255 with optional percentile clip + gamma.

    clip_hi_pct: if set (e.g. 99.0), override mx with that percentile of `a`.
    gamma: <1 brightens low-end detail; >1 darkens it. 1.0 = linear.
    """
    if clip_hi_pct is not None and len(a):
        mx = float(np.percentile(a, clip_hi_pct))
    norm = np.clip((a - mn) / max(mx - mn, 1e-9), 0, 1)
    if gamma != 1.0:
        norm = norm ** gamma
    return (norm * 255).astype(np.uint8)


def front_view(points, val="depth", h_res=0.2, v_res=0.2,
               h_fov=(-60, 60), v_fov=(-15, 15), y_fudge=3,
               gamma=1.0, clip_hi_pct=None):
    """Angular front-view projection of Nx6 lidar frame to HxW uint8."""
    pts = _filter_nonzero(points)
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    x, y, z, refl = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    d_xy = np.sqrt(x ** 2 + y ** 2)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, d_xy))
    xi = ((az - h_fov[0]) / h_res).astype(np.int32)
    yi = ((el - v_fov[0]) / v_res).astype(np.int32)
    w = int(np.ceil((h_fov[1] - h_fov[0]) / h_res))
    h = int(np.ceil((v_fov[1] - v_fov[0]) / v_res)) + y_fudge
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi, yi, refl, z_v, d_v = xi[ok], yi[ok], refl[ok], z[ok], d_xy[ok]
    if val == "reflectance":
        pv, mn, mx = refl, 0, max(refl.max() if refl.size else 1, 1)
    elif val == "height":
        pv, mn, mx = z_v, z_v.min() if z_v.size else 0, z_v.max() if z_v.size else 1
    else:
        pv, mn, mx = d_v, d_v.min() if d_v.size else 0, d_v.max() if d_v.size else 1
    img = np.zeros((h, w), dtype=np.uint8)
    if xi.size:
        img[h - 1 - yi, xi] = _scale_to_255(pv, mn, mx, gamma, clip_hi_pct)
    return img


def birds_eye_view(points, side_range=(-10, 10), fwd_range=(0, 20),
                   res=0.05, min_height=-3.0, max_height=2.0,
                   gamma=1.0, clip_hi_pct=None):
    """Top-down BEV projection, pixel = height-scaled."""
    pts = _filter_nonzero(points)
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ok = ((x > fwd_range[0]) & (x < fwd_range[1]) &
          (y > side_range[0]) & (y < side_range[1]))
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    xi = ((y - side_range[0]) / res).astype(np.int32)
    yi = ((x - fwd_range[0]) / res).astype(np.int32)
    w = int((side_range[1] - side_range[0]) / res)
    h = int((fwd_range[1] - fwd_range[0]) / res)
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    pv = _scale_to_255(np.clip(z, min_height, max_height), min_height, max_height,
                       gamma, clip_hi_pct)
    img = np.zeros((h, w), dtype=np.uint8)
    img[h - 1 - yi, xi] = pv
    return img


def panorama(points, h_res=0.2, v_res=0.2, v_fov=(-15, 15),
             d_range=(0, 30), y_fudge=3, gamma=1.0, clip_hi_pct=None):
    """Spherical panorama depth projection."""
    pts = _filter_nonzero(points)
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    d_xy = np.sqrt(x ** 2 + y ** 2)
    xi = np.arctan2(y, x) / np.radians(h_res)
    yi = -(np.arctan2(z, d_xy) / np.radians(v_res))
    x_min = -60.0 / h_res
    xi = np.trunc(-xi - x_min).astype(np.int32)
    x_max = int(np.ceil(120.0 / h_res))
    y_min = -(v_fov[1] / v_res + y_fudge)
    yi = np.trunc(yi - y_min).astype(np.int32)
    y_max = int(np.ceil((v_fov[1] - v_fov[0]) / v_res)) + y_fudge
    ok = (xi >= 0) & (xi < x_max) & (yi >= 0) & (yi < y_max)
    xi, yi = xi[ok], yi[ok]
    d_xy = np.clip(d_xy[ok], d_range[0], d_range[1])
    img = np.zeros((y_max, x_max), dtype=np.uint8)
    if xi.size:
        img[yi, xi] = _scale_to_255(d_xy, d_range[0], d_range[1], gamma, clip_hi_pct)
    return img


def _index(path: Path) -> int:
    m = _IDX_RE.search(path.stem)
    if not m:
        raise ValueError(f"no numeric index in {path.name}")
    return int(m.group(1))


def pair_files(run_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(image_path, lidar_path), ...] paired by filename index."""
    imgs = {_index(p): p for p in sorted((run_dir / "images").glob("*.jpg"))}
    lids = {_index(p): p for p in sorted((run_dir / "lidar").glob("*.npy"))}
    common = sorted(set(imgs) & set(lids))
    return [(imgs[i], lids[i]) for i in common]


def render_lidar(frame: np.ndarray, view: str,
                 gamma: float = 1.0, clip_hi_pct: float | None = None,
                ) -> tuple[np.ndarray, str]:
    """Project Nx6 lidar frame to 2D image; return (img, cmap).

    gamma<1 boosts low-end detail. clip_hi_pct (e.g. 99) compresses bright outliers.
    """
    kw = dict(gamma=gamma, clip_hi_pct=clip_hi_pct)
    if view == "front":
        return front_view(frame, val="depth", **kw), "jet"
    if view == "pano":
        return panorama(frame, **kw), "jet"
    if view == "bev":
        return birds_eye_view(frame, side_range=(-10, 10), fwd_range=(0, 20), **kw), "gray"
    raise ValueError(f"unknown view: {view}")


def plot_pairs(pairs, view="front", save=None):
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    for r, (img_path, lid_path) in enumerate(pairs):
        img = imread(img_path)
        pts = np.load(lid_path)
        lidar_img, cmap = render_lidar(pts, view)

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"webcam: {img_path.name}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(lidar_img, cmap=cmap)
        axes[r, 1].set_title(f"lidar ({view}): {lid_path.name}  [{pts.shape[0]} pts]")
        axes[r, 1].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120)
        print(f"saved {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--index", "-i", type=int, help="plot single frame by index")
    ap.add_argument("--stride", "-s", type=int, default=1, help="every Nth frame")
    ap.add_argument("--limit", "-n", type=int, default=6, help="max frames")
    ap.add_argument("--view", choices=["front", "pano", "bev"], default="front")
    ap.add_argument("--save", type=Path, help="save to PNG instead of showing")
    args = ap.parse_args()

    pairs = pair_files(args.run_dir)
    if not pairs:
        raise SystemExit(f"no paired files in {args.run_dir}")

    if args.index is not None:
        pairs = [p for p in pairs if _index(p[0]) == args.index]
    else:
        pairs = pairs[:: args.stride][: args.limit]

    print(f"plotting {len(pairs)} pair(s) from {args.run_dir}")
    plot_pairs(pairs, view=args.view, save=args.save)


if __name__ == "__main__":
    main()
