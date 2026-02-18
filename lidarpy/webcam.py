"""Webcam image collection for USB cameras (e.g. Logitech BRIO 046d:085e)."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2


def find_video_device(usb_id: str = "046d:085e") -> int | None:
    """Find /dev/videoN index for a USB camera by vendor:product ID.

    Walks /sys/class/video4linux to match against the USB device tree.
    Returns the lowest video index whose parent USB device matches *usb_id*,
    or None if not found.
    """
    vid, pid = usb_id.lower().split(":")
    v4l_root = Path("/sys/class/video4linux")
    if not v4l_root.exists():
        return None

    candidates: list[int] = []
    for entry in sorted(v4l_root.iterdir()):
        # Follow symlink to real sysfs path, walk up looking for idVendor/idProduct
        real = entry.resolve()
        cur = real
        while cur != cur.parent:
            id_vendor = cur / "idVendor"
            id_product = cur / "idProduct"
            if id_vendor.exists() and id_product.exists():
                v = id_vendor.read_text().strip().lower()
                p = id_product.read_text().strip().lower()
                if v == vid and p == pid:
                    m = re.search(r"video(\d+)", entry.name)
                    if m:
                        candidates.append(int(m.group(1)))
                break
            cur = cur.parent

    return min(candidates) if candidates else None


def open_camera(device: int, width: int = 1920, height: int = 1080) -> cv2.VideoCapture:
    """Open a V4L2 capture device and set resolution."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # let auto-exposure settle
    for _ in range(5):
        cap.read()
    return cap


def capture_loop(
    cap: cv2.VideoCapture,
    out_dir: Path,
    interval: float = 1.0,
    count: int = 0,
    fmt: str = "jpg",
    prefix: str = "img",
) -> int:
    """Capture images in a loop.

    Args:
        cap: opened VideoCapture
        out_dir: directory to save images
        interval: seconds between captures (0 = as fast as possible)
        count: number of images to capture (0 = unlimited, Ctrl-C to stop)
        fmt: image format extension (jpg, png, bmp)
        prefix: filename prefix

    Returns:
        number of images captured
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        while count == 0 or n < count:
            ret, frame = cap.read()
            if not ret:
                print("frame grab failed, retrying...")
                time.sleep(0.1)
                continue
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = out_dir / f"{prefix}_{ts}.{fmt}"
            cv2.imwrite(str(fname), frame)
            n += 1
            print(f"[{n}] {fname.name}  ({frame.shape[1]}x{frame.shape[0]})")
            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nstopped after {n} images")
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Capture images from USB webcam")
    p.add_argument("-d", "--device", type=int, default=None,
                   help="/dev/videoN index (auto-detect BRIO if omitted)")
    p.add_argument("-u", "--usb-id", default="046d:085e",
                   help="USB vendor:product to auto-detect (default: 046d:085e Logitech BRIO)")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("captures"),
                   help="output directory (default: captures/)")
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="seconds between captures (default: 1.0)")
    p.add_argument("-n", "--count", type=int, default=0,
                   help="number of images (0 = unlimited, Ctrl-C to stop)")
    p.add_argument("-W", "--width", type=int, default=1920,
                   help="capture width (default: 1920)")
    p.add_argument("-H", "--height", type=int, default=1080,
                   help="capture height (default: 1080)")
    p.add_argument("-f", "--format", default="jpg", choices=["jpg", "png", "bmp"],
                   help="image format (default: jpg)")
    p.add_argument("--prefix", default="img", help="filename prefix (default: img)")
    args = p.parse_args(argv)

    device = args.device
    if device is None:
        print(f"auto-detecting USB camera {args.usb_id} ...")
        device = find_video_device(args.usb_id)
        if device is None:
            raise SystemExit(f"no video device found for USB ID {args.usb_id}")
        print(f"found /dev/video{device}")

    cap = open_camera(device, args.width, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"capturing {actual_w}x{actual_h} -> {args.out_dir}/")

    try:
        total = capture_loop(cap, args.out_dir, args.interval, args.count,
                             args.format, args.prefix)
        print(f"done, {total} images saved to {args.out_dir}/")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
