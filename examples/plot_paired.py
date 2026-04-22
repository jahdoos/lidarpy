"""Plot paired webcam image + lidar 2D projection from an acquisition folder.

Usage:
    python examples/plot_paired.py <run_dir> [--index N] [--stride S]
                                   [--save OUT.png] [--view front|pano|bev]

<run_dir> must contain 'images/' (*.jpg) and 'lidar/' (*.npy) with paired
indices like frame_000042.jpg ↔ points_000042.npy.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lidar_projection_example import birds_eye_view, front_view, panorama  # noqa: E402

_IDX_RE = re.compile(r"(\d+)")


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


def render_lidar(frame: np.ndarray, view: str) -> tuple[np.ndarray, str]:
    """Project Nx6 lidar frame to 2D image; return (img, cmap)."""
    if view == "front":
        return front_view(frame, val="depth"), "jet"
    if view == "pano":
        return panorama(frame), "jet"
    if view == "bev":
        return birds_eye_view(frame, side_range=(-10, 10), fwd_range=(0, 20)), "gray"
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
