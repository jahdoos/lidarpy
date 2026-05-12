"""Temporal + spatial alignment for paired webcam / lidar sequences.

Streams differ in modality (RGB vs depth/reflectance), FOV, aspect ratio, and
sample rate. Alignment strategy:

- Time: cross-correlate per-frame motion magnitudes (mean |I[t] - I[t-1]|).
  Modality-invariant — both streams see the same scene motion.

- Space: scale is computed analytically from known FOVs:
      sx = lidar_hfov_deg / webcam_hfov_deg
      sy = lidar_vfov_deg / webcam_vfov_deg
  Translation is estimated by phase-correlating median-background edge maps,
  cropped to the overlap region (so zero-padding outside each FOV doesn't
  dominate the FFT).

Public API:
    align_sequences(webcam_frames, lidar_frames, ...) -> Alignment
    apply_spatial(lidar_img, target_hw, alignment) -> ndarray
    estimate_time_lag / estimate_spatial_shift  (lower-level)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Alignment:
    time_lag: int                     # webcam[k] matches lidar[k + time_lag]
    scale: tuple[float, float]        # (sx, sy) — multiply canvas dims
    shift: tuple[float, float]        # (dx, dy) — canvas pixels, post-scale
    score: float                      # phase-corr peak (higher = better)
    canvas: tuple[int, int]           # (H, W) canvas used during fit


# --- preprocessing ----------------------------------------------------------

def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32)


def _resize_gray(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    H, W = hw
    return cv2.resize(_to_gray_f32(img), (W, H))


def _edge_mag(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mx = float(mag.max())
    return mag / mx if mx > 0 else mag


def _median_bg(frames, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Per-pixel median grayscale across frames — rejects moving content.

    If target_hw is None, frames with a shape different from the majority are
    dropped (handles render fallbacks like empty-point 10x10 stubs).
    """
    if not frames:
        return np.zeros((2, 2), dtype=np.float32)
    if target_hw is not None:
        stack = np.stack([_resize_gray(f, target_hw) for f in frames], axis=0)
    else:
        shapes = [f.shape[:2] for f in frames]
        # majority shape
        from collections import Counter
        top = Counter(shapes).most_common(1)[0][0]
        kept = [_to_gray_f32(f) for f in frames if f.shape[:2] == top]
        if not kept:
            return np.zeros((2, 2), dtype=np.float32)
        stack = np.stack(kept, axis=0)
    return np.median(stack, axis=0).astype(np.float32)


# --- temporal --------------------------------------------------------------

def _motion_signal(frames, canvas: tuple[int, int] = (90, 160)) -> np.ndarray:
    """Per-frame mean |I[t] - I[t-1]| on small gray images. Length = N - 1."""
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float32)
    g = [_resize_gray(f, canvas) for f in frames]
    return np.array(
        [float(np.abs(g[i] - g[i - 1]).mean()) for i in range(1, len(g))],
        dtype=np.float32,
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def estimate_time_lag(webcam_frames, lidar_frames, max_lag: int = 10
                      ) -> tuple[int, float]:
    """Find integer L such that webcam[k] best matches lidar[k + L].

    Returns (lag, score). Positive L → lidar is delayed vs webcam.
    """
    a = _zscore(_motion_signal(webcam_frames))
    b = _zscore(_motion_signal(lidar_frames))
    if len(a) == 0 or len(b) == 0:
        return 0, 0.0
    # np.correlate(a, b)[k'] ≈ Σ a[n+k] · b[n], k = k' - (len(a)-1).
    # Peak at k means a[n+k] ≈ b[n] → webcam[m] ≈ lidar[m - k] → L = -k.
    corr = np.correlate(a, b, mode="full") / len(a)
    lags = np.arange(-len(a) + 1, len(a))
    mask = np.abs(lags) <= max_lag
    idx = int(np.argmax(corr[mask]))
    return -int(lags[mask][idx]), float(corr[mask][idx])


# --- spatial ---------------------------------------------------------------

def _overlap_roi(H: int, W: int, lh: int, lw: int) -> tuple[int, int, int, int]:
    """Rows/cols in canvas covered by a centered image of size (lh, lw)."""
    y0 = max(0, (H - lh) // 2); y1 = min(H, y0 + min(lh, H))
    x0 = max(0, (W - lw) // 2); x1 = min(W, x0 + min(lw, W))
    return y0, y1, x0, x1


def _center_place(src: np.ndarray, dst_hw: tuple[int, int]) -> np.ndarray:
    """Center-crop or zero-pad src to dst_hw."""
    H, W = dst_hw
    h, w = src.shape
    out = np.zeros(dst_hw, dtype=src.dtype)
    sh, sw = min(h, H), min(w, W)
    y0s, x0s = (h - sh) // 2, (w - sw) // 2
    y0d, x0d = (H - sh) // 2, (W - sw) // 2
    out[y0d:y0d + sh, x0d:x0d + sw] = src[y0s:y0s + sh, x0s:x0s + sw]
    return out


def estimate_spatial_shift(
    webcam_frames, lidar_frames,
    sx: float, sy: float,
    canvas: tuple[int, int] = (360, 640),
) -> tuple[float, float, float]:
    """Phase-correlate edge maps in overlap region to recover (dx, dy).

    Returns (dx, dy, score). Scale (sx, sy) MUST be provided (from FOVs).
    """
    H, W = canvas

    # Webcam bg + edges at canvas size
    w_bg = _median_bg(webcam_frames, canvas)
    w_edge = _edge_mag(w_bg)

    # Lidar median at native size, then scale → canvas
    l_native = _median_bg(lidar_frames)
    lh = max(2, int(round(H * sy)))
    lw = max(2, int(round(W * sx)))
    l_scaled = cv2.resize(l_native, (lw, lh))
    l_edges_full = _edge_mag(l_scaled)
    l_fit = _center_place(l_edges_full, canvas).astype(np.float32)

    # Restrict both to overlap ROI → avoid zero-padding FFT artifacts
    y0, y1, x0, x1 = _overlap_roi(H, W, lh, lw)
    w_roi = w_edge[y0:y1, x0:x1].astype(np.float32)
    l_roi = l_fit[y0:y1, x0:x1].astype(np.float32)
    if w_roi.shape[0] < 4 or w_roi.shape[1] < 4:
        return 0.0, 0.0, 0.0

    hann = cv2.createHanningWindow((w_roi.shape[1], w_roi.shape[0]), cv2.CV_32F)
    (dx, dy), score = cv2.phaseCorrelate(l_roi, w_roi, hann)
    return float(dx), float(dy), float(score)


# --- high-level ------------------------------------------------------------

def align_sequences(
    webcam_frames, lidar_frames,
    lidar_fov_deg: tuple[float, float] = (120.0, 30.0),   # HAP H, V
    webcam_fov_deg: tuple[float, float] = (70.0, 43.0),   # typical USB webcam H, V
    max_lag: int = 10,
    canvas: tuple[int, int] = (360, 640),
) -> Alignment:
    """Estimate temporal lag + spatial transform.

    FOVs fix the scale analytically (sx = lidar_h / webcam_h, sy = lidar_v / webcam_v).
    Only translation is searched from imagery. HAP default FOV is 120° × 30°;
    measure/override webcam FOV for your camera.

    Args:
        webcam_frames: list of HxWx3 (or HxW) uint8 frames.
        lidar_frames:  list of HxW uint8 2D projections, same indexing as webcam_frames.
        lidar_fov_deg:  (H, V) in degrees.
        webcam_fov_deg: (H, V) in degrees.
        max_lag:       max |frame shift| to consider for temporal search.
        canvas:        (H, W) canvas used during fit.

    Returns: Alignment(time_lag, scale, shift, score, canvas).
    """
    lag, _ = estimate_time_lag(webcam_frames, lidar_frames, max_lag=max_lag)
    if lag >= 0:
        w = webcam_frames[: len(webcam_frames) - lag]
        l = lidar_frames[lag:]
    else:
        w = webcam_frames[-lag:]
        l = lidar_frames[: len(lidar_frames) + lag]
    n = min(len(w), len(l))
    w, l = w[:n], l[:n]

    sx = lidar_fov_deg[0] / webcam_fov_deg[0]
    sy = lidar_fov_deg[1] / webcam_fov_deg[1]
    dx, dy, score = estimate_spatial_shift(w, l, sx, sy, canvas=canvas)
    return Alignment(time_lag=lag, scale=(sx, sy), shift=(dx, dy),
                     score=score, canvas=canvas)


def apply_spatial(
    lidar_img: np.ndarray,
    target_hw: tuple[int, int],
    alignment: Alignment,
) -> np.ndarray:
    """Warp lidar_img into target_hw using alignment computed at alignment.canvas."""
    H, W = target_hw
    cH, cW = alignment.canvas
    sx, sy = alignment.scale
    dx, dy = alignment.shift

    lw_c = max(2, int(round(cW * sx)))
    lh_c = max(2, int(round(cH * sy)))
    fx, fy = W / cW, H / cH
    out_w = int(round(lw_c * fx))
    out_h = int(round(lh_c * fy))

    scaled = cv2.resize(lidar_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    tx = (W - out_w) / 2 + dx * fx
    ty = (H - out_h) / 2 + dy * fy
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        scaled, M, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
