"""Render HAP LiDAR frames using lidar_projection-style techniques.

Adapts front-view, bird's-eye, and panorama projections from
github.com/collector-m/lidar_projection for the Livox HAP's
forward-facing FOV and Nx6 frame format [x,y,z,refl,tag,ts_us].

No external deps beyond numpy + matplotlib (for display/save).
"""

import numpy as np
import matplotlib.pyplot as plt


# -- helpers ------------------------------------------------------------------

def scale_to_255(a, mn, mx):
    """Scale array from [mn, mx] to [0, 255] uint8."""
    return np.clip(((a - mn) / (mx - mn)) * 255, 0, 255).astype(np.uint8)


def _filter_nonzero(points):
    """Remove zero-distance (no-return) points."""
    dist_sq = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
    return points[dist_sq > 0]


# -- front view (angular projection) -----------------------------------------

def front_view(points, val="reflectance", h_res=0.2, v_res=0.2,
               h_fov=(-60, 60), v_fov=(-15, 15), y_fudge=3):
    """Project point cloud to 2D front-view image via angular binning.

    Adapted from lidar_projection's lidar_to_2d_front_view() for HAP coords:
      X = forward (+X = away from sensor)
      Y = left/right
      Z = up/down

    Args:
        points: Nx6 frame [x, y, z, reflectivity, tag, timestamp_us].
        val: Pixel value — "reflectance", "depth", or "height".
        h_res: Horizontal angular resolution (degrees/pixel).
        v_res: Vertical angular resolution (degrees/pixel).
        h_fov: (min, max) horizontal FOV in degrees. HAP ~±60°.
        v_fov: (min, max) vertical FOV in degrees. HAP ~±12.5°.
        y_fudge: Extra rows to add to image height.

    Returns:
        HxW uint8 image.
    """
    pts = _filter_nonzero(points)
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    x, y, z, refl = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    d_xy = np.sqrt(x ** 2 + y ** 2)

    # Angular coordinates (HAP: X forward, Y lateral, Z vertical)
    azimuth = np.degrees(np.arctan2(y, x))       # horiz angle from +X
    elevation = np.degrees(np.arctan2(z, d_xy))   # vert angle from XY plane

    # Map to pixel coords
    h_fov_total = h_fov[1] - h_fov[0]
    v_fov_total = v_fov[1] - v_fov[0]

    x_img = ((azimuth - h_fov[0]) / h_res).astype(np.int32)
    y_img = ((elevation - v_fov[0]) / v_res).astype(np.int32)

    w = int(np.ceil(h_fov_total / h_res))
    h = int(np.ceil(v_fov_total / v_res)) + y_fudge

    # Clip to image bounds
    valid = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    x_img, y_img = x_img[valid], y_img[valid]
    refl, z_v, d_v = refl[valid], z[valid], d_xy[valid]

    # Choose pixel value
    if val == "reflectance":
        pv = refl
        mn, mx = 0, max(refl.max(), 1)
    elif val == "height":
        pv = z_v
        mn, mx = z_v.min(), z_v.max()
    else:  # depth
        pv = d_v
        mn, mx = d_v.min(), d_v.max()

    img = np.zeros((h, w), dtype=np.uint8)
    img[h - 1 - y_img, x_img] = scale_to_255(pv, mn, mx)  # flip so +Z = top
    return img


# -- bird's eye view ---------------------------------------------------------

def birds_eye_view(points, side_range=(-10, 10), fwd_range=(0, 20),
                   res=0.05, min_height=-3.0, max_height=2.0):
    """Top-down bird's-eye view, pixel = height scaled to 0-255.

    Adapted from lidar_projection's birds_eye_point_cloud() for HAP coords:
      X = forward → image Y (top=far)
      Y = lateral → image X

    Args:
        points: Nx6 frame.
        side_range: (left, right) lateral bounds in meters.
        fwd_range: (near, far) forward bounds in meters.
        res: Meters per pixel.
        min_height, max_height: Z clamp range.

    Returns:
        HxW uint8 image.
    """
    pts = _filter_nonzero(points)
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Filter to ROI
    valid = ((x > fwd_range[0]) & (x < fwd_range[1]) &
             (y > side_range[0]) & (y < side_range[1]))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    # Map to pixel coords
    x_img = ((y - side_range[0]) / res).astype(np.int32)
    y_img = ((x - fwd_range[0]) / res).astype(np.int32)

    w = int((side_range[1] - side_range[0]) / res)
    h = int((fwd_range[1] - fwd_range[0]) / res)

    # Clip
    x_img = np.clip(x_img, 0, w - 1)
    y_img = np.clip(y_img, 0, h - 1)

    pv = np.clip(z, min_height, max_height)
    pv = scale_to_255(pv, min_height, max_height)

    img = np.zeros((h, w), dtype=np.uint8)
    img[h - 1 - y_img, x_img] = pv  # flip so far = top
    return img


# -- panorama (spherical projection) -----------------------------------------

def panorama(points, h_res=0.2, v_res=0.2, v_fov=(-15, 15),
             d_range=(0, 30), y_fudge=3):
    """Spherical projection to panoramic depth image.

    Adapted from lidar_projection's point_cloud_to_panorama() for HAP FOV.

    Args:
        points: Nx6 frame.
        h_res, v_res: Angular resolution in degrees.
        v_fov: Vertical FOV (min, max) in degrees.
        d_range: Depth clamp range in meters.
        y_fudge: Extra image rows.

    Returns:
        HxW uint8 depth image.
    """
    pts = _filter_nonzero(points)
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    d_xy = np.sqrt(x ** 2 + y ** 2)

    v_fov_total = v_fov[1] - v_fov[0]

    # Spherical mapping
    x_img = np.arctan2(y, x) / np.radians(h_res)
    y_img = -(np.arctan2(z, d_xy) / np.radians(v_res))

    # Image dimensions from HAP's actual horizontal FOV (~120°)
    x_min = -60.0 / h_res
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(120.0 / h_res))  # ~120° total horiz FOV

    y_min = -(v_fov[1] / v_res + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)
    y_max = int(np.ceil(v_fov_total / v_res)) + y_fudge

    # Clip
    valid = ((x_img >= 0) & (x_img < x_max) &
             (y_img >= 0) & (y_img < y_max))
    x_img, y_img = x_img[valid], y_img[valid]
    d_xy = np.clip(d_xy[valid], d_range[0], d_range[1])

    img = np.zeros((y_max, x_max), dtype=np.uint8)
    img[y_img, x_img] = scale_to_255(d_xy, d_range[0], d_range[1])
    return img


# -- main: load a frame and render all views ---------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/run1/lidar/points_000010.npy"
    frame = np.load(path)
    print(f"Loaded {path}: {frame.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Front view — reflectance
    img_refl = front_view(frame, val="reflectance")
    axes[0, 0].imshow(img_refl, cmap="gray")
    axes[0, 0].set_title("Front view (reflectance)")

    # 2. Front view — depth
    img_depth = front_view(frame, val="depth")
    axes[0, 1].imshow(img_depth, cmap="jet")
    axes[0, 1].set_title("Front view (depth)")

    # 3. Bird's eye view
    img_bev = birds_eye_view(frame, side_range=(-10, 2), fwd_range=(0, 14))
    axes[1, 0].imshow(img_bev, cmap="gray")
    axes[1, 0].set_title("Bird's eye view (height)")

    # 4. Panorama depth
    img_pano = panorama(frame)
    axes[1, 1].imshow(img_pano, cmap="jet")
    axes[1, 1].set_title("Panorama (depth)")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("data/lidar_projections.png", dpi=150)
    plt.show()
    print("Saved data/lidar_projections.png")
