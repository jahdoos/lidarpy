import os
import time
import argparse
import datetime
import threading
from lidarpy.webcam import find_video_device, open_camera
import numpy as np
import cv2
from lidarpy.csdk import CsdkLidar

#!/usr/bin/env python3
"""run_acquisition.py — collect CsdkLidar points + Logitech Brio images."""

def init_camera(index, width=None, height=None):
    if cv2 is None:
        return None
    try:
        dev = find_video_device(index)
        if dev is None:
            print(f"camera USB device {index!r} not found")
            return None
        return open_camera(dev)
    except Exception as e:
        print(f"camera init skipped: {e}")
        return None


def init_lidar(**kwargs):
    if CsdkLidar is None:
        return None
    try:
        lidar = CsdkLidar(**kwargs)
        if hasattr(lidar, "connect"):
            lidar.connect()
        if hasattr(lidar, "start"):
            lidar.start()
            return lidar
    except Exception as e:
        print(f"init_lidar error: {e}")
        return None


def lidar_reader(lidar, buffer, lock, stop_event):
    """Continuously read lidar frames into a shared buffer."""
    while not stop_event.is_set():
        try:
            pts = lidar.get_frame()
        except Exception as e:
            print(f"get_frame error: {e}")
            pts = None
        if pts is not None and pts.shape[0] > 0:
            with lock:
                buffer.append(pts)


def collect(args):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = os.path.abspath(args.out_dir or f"acq_{ts}")
    img_dir = os.path.join(out, "images")
    lidar_dir = os.path.join(out, "lidar")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)

    cap = None
    if not args.no_camera:
        cap = init_camera(args.camera_index)
        if cap is None:
            print("no-camera — continuing with lidar only")
    try:
        lidar = init_lidar(config_path=args.config_path, host_ip=args.host_ip, sdk_lib_path=args.sdk_lib_path)
        if lidar is not None:
            time.sleep(1)
            lidar.get_frame()  # flush buffer
    except Exception as e:
        print(f"init_lidar error: {e}")
        lidar = None

    stop_event = threading.Event()
    lidar_thread = None
    lidar_buffer = []
    lidar_lock = threading.Lock()
    if lidar is not None:
        lidar_thread = threading.Thread(
            target=lidar_reader,
            args=(lidar, lidar_buffer, lidar_lock, stop_event),
            daemon=True,
        )
        lidar_thread.start()

    i = 0
    start = time.time()
    try:
        while True:
            if args.duration and (time.time() - start) >= args.duration:
                break
            t = time.time()
            if cap:
                ret, frame = cap.read()
                if ret and frame is not None:
                    fname = os.path.join(img_dir, f"frame_{i:06d}.jpg")
                    cv2.imwrite(fname, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if lidar is not None:
                with lidar_lock:
                    chunks = lidar_buffer.copy()
                    lidar_buffer.clear()
                if chunks:
                    merged = np.vstack(chunks)
                    merged = merged[~(merged[:, :5].sum(axis=1) == 0)]
                    if merged.shape[0] > 0:
                        np.save(os.path.join(lidar_dir, f"points_{i:06d}.npy"), merged)
            i += 1
            if args.fps and args.fps > 0:
                time.sleep(max(0, 1.0 / args.fps - (time.time() - t)))
            elif not cap and lidar is None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if lidar_thread is not None:
            lidar_thread.join(timeout=2.0)
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        if lidar:
            try:
                if hasattr(lidar, "stop"):
                    lidar.stop()
                if hasattr(lidar, "close"):
                    lidar.close()
            except Exception:
                pass
    print(out)


def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--duration", "-d", type=float, default=0.0,
                   help="seconds to run (0 = until Ctrl-C)")
    p.add_argument("-out_dir", "-o", default=None, help="output dir")
    p.add_argument("--camera_index", type=str, default="046d:085e", help="cv2 camera index")
    # p.add_argument("--cam-width", type=int, default=1920, help="camera width")
    # p.add_argument("--cam-height", type=int, default=1080, help="camera height")
    p.add_argument("--fps", type=float, default=30.0, help="capture rate — camera and lidar save once per tick")
    p.add_argument("--no-camera", action="store_true", help="skip webcam, lidar only")
    p.add_argument("--host_ip", default="192.168.100.5",
                    help="CsdkLidar host ip")
    p.add_argument("--sdk_lib_path",
                    default="/home/rfor10/Livox-SDK2/build/sdk_core/liblivox_lidar_sdk_shared.so", help = "path to built Livox SDK2 shared library")
    p.add_argument("--config_path", default = "hap_config.json", help="config JSON file for SDK initialization")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args)