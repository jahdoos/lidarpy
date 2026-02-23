"""Synchronized lidar + webcam acquisition.

Initializes a CsdkLidar and USB webcam, then collects paired
(point-cloud frame, camera image) snapshots at a configurable interval.

Usage as script::

    python -m lidarpy.collect -c config.json -o data/ -n 10 -i 0.5

Usage from Python::

    from lidarpy.collect import Collector

    with Collector("config.json") as c:
        pairs = c.collect(count=5, interval=1.0)
        # pairs is a list of (timestamp, np.ndarray Nx6, np.ndarray HxWx3)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lidarpy.csdk import CsdkLidar
from lidarpy.webcam import find_video_device, open_camera


class Collector:
    """Synchronized CsdkLidar + webcam acquisition.

    Parameters:
        config_path: path to Livox SDK2 JSON config file
        host_ip: host NIC address (empty = auto from config)
        device: /dev/videoN index; auto-detected from *usb_id* when None
        usb_id: USB vendor:product for webcam auto-detection
        width: camera capture width
        height: camera capture height
        lidar_timeout: seconds to wait for lidar device discovery
    """

    def __init__(
        self,
        config_path: str,
        host_ip: str = "",
        device: int | None = None,
        usb_id: str = "046d:085e",
        width: int = 1920,
        height: int = 1080,
        lidar_timeout: float = 10.0,
    ):
        self._config_path = config_path
        self._host_ip = host_ip
        self._usb_id = usb_id
        self._width = width
        self._height = height
        self._lidar_timeout = lidar_timeout

        if device is None:
            device = find_video_device(usb_id)
            if device is None:
                raise RuntimeError(f"no video device for USB ID {usb_id}")
        self._device = device

        self._lidar: CsdkLidar | None = None
        self._cap: cv2.VideoCapture | None = None

    # -- lifecycle --

    def open(self) -> None:
        """Initialize both sensors and start lidar sampling."""
        self._lidar = CsdkLidar(self._config_path, self._host_ip)
        self._lidar.connect(timeout=self._lidar_timeout)
        self._lidar.start()
        self._cap = open_camera(self._device, self._width, self._height)

    def close(self) -> None:
        """Stop lidar and release webcam."""
        if self._lidar is not None:
            self._lidar.close()
            self._lidar = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- acquisition --

    def grab(self, frame_timeout: float = 1.0) -> tuple[str, np.ndarray, np.ndarray]:
        """Acquire one synchronized (lidar frame, camera image) pair.

        The webcam image is captured immediately before the lidar frame
        so both represent approximately the same instant.

        Returns:
            (timestamp_str, points Nx6, image HxWx3)
        """
        if self._lidar is None or self._cap is None:
            raise RuntimeError("call open() first")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        ret, image = self._cap.read()
        if not ret:
            raise RuntimeError("webcam frame grab failed")

        points = self._lidar.get_frame(timeout=frame_timeout)

        return ts, points, image

    def collect(
        self,
        count: int = 1,
        interval: float = 1.0,
        frame_timeout: float = 1.0,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Collect *count* synchronized pairs at *interval* seconds apart.

        Returns:
            list of (timestamp_str, points Nx6, image HxWx3)
        """
        pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
        for i in range(count):
            ts, pts, img = self.grab(frame_timeout=frame_timeout)
            pairs.append((ts, pts, img))
            print(f"[{i + 1}/{count}] {ts}  pts={pts.shape[0]}  img={img.shape[1]}x{img.shape[0]}")
            if i < count - 1 and interval > 0:
                time.sleep(interval)
        return pairs

    def collect_to_disk(
        self,
        out_dir: Path,
        count: int = 1,
        interval: float = 1.0,
        frame_timeout: float = 1.0,
        img_fmt: str = "jpg",
    ) -> int:
        """Collect pairs and save to *out_dir*.

        Lidar frames saved as ``lidar_{ts}.npy``, images as ``img_{ts}.{fmt}``.

        Returns:
            number of pairs saved
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        try:
            for i in range(count if count > 0 else 2**63):
                ts, pts, img = self.grab(frame_timeout=frame_timeout)
                np.save(str(out_dir / f"lidar_{ts}.npy"), pts)
                cv2.imwrite(str(out_dir / f"img_{ts}.{img_fmt}"), img)
                n += 1
                print(
                    f"[{n}] {ts}  pts={pts.shape[0]}  "
                    f"img={img.shape[1]}x{img.shape[0]}"
                )
                if count > 0 and n >= count:
                    break
                if interval > 0:
                    time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\nstopped after {n} pairs")
        return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Synchronized lidar + webcam acquisition"
    )
    p.add_argument("-c", "--config", required=True,
                   help="Livox SDK2 config JSON path")
    p.add_argument("--host-ip", default="",
                   help="host NIC address (default: from config)")
    p.add_argument("-d", "--device", type=int, default=None,
                   help="/dev/videoN index (auto-detect if omitted)")
    p.add_argument("-u", "--usb-id", default="046d:085e",
                   help="USB vendor:product for auto-detect (default: BRIO)")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("data"),
                   help="output directory (default: data/)")
    p.add_argument("-n", "--count", type=int, default=0,
                   help="number of pairs (0 = unlimited, Ctrl-C to stop)")
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="seconds between captures (default: 1.0)")
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    p.add_argument("-f", "--format", default="jpg",
                   choices=["jpg", "png", "bmp"])
    p.add_argument("--lidar-timeout", type=float, default=10.0,
                   help="lidar discovery timeout in seconds")
    args = p.parse_args(argv)

    with Collector(
        config_path=args.config,
        host_ip=args.host_ip,
        device=args.device,
        usb_id=args.usb_id,
        width=args.width,
        height=args.height,
        lidar_timeout=args.lidar_timeout,
    ) as c:
        n = c.collect_to_disk(
            out_dir=args.out_dir,
            count=args.count,
            interval=args.interval,
            img_fmt=args.format,
        )
        print(f"done, {n} pairs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
