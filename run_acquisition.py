import os
import time
import argparse
import datetime
from lidarpy.webcam import find_video_device, open_camera
import numpy as np
import cv2
from lidarpy.csdk import CsdkLidar

#!/usr/bin/env python3
"""run_acquisition.py — collect CsdkLidar points + Logitech Brio images."""

def init_camera(index, width=None, height=None):
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    dev = find_video_device(index)
    cap = open_camera(dev)
    if not cap or not cap.isOpened():
        return None
    return cap


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


def collect(args):
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = os.path.abspath(args.out_dir or f"acq_{ts}")
    img_dir = os.path.join(out, "images")
    lidar_dir = os.path.join(out, "lidar")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)

    cap = init_camera(args.camera_index)
    if cap is None:
        print("no-camera")
    try:
        lidar = init_lidar(config_path=args.config_path, host_ip=args.host_ip, sdk_lib_path=args.sdk_lib_path)
        time.sleep(1)
        lidar.get_frame(); # flush buffer
    except Exception as e:
        print(f"init_lidar error: {e}")
        lidar = None

    i = 0
    start = time.time()
    try:
        while True:
            if args.duration and (time.time() - start) >= args.duration:
                break
            t = time.time()
            # image
            frame = None
            if cap:
                ret, frame = cap.read()
                if ret and frame is not None:
                    fname = os.path.join(img_dir, f"frame_{i:06d}.jpg")
                    cv2.imwrite(fname, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # lidar
            if lidar:
                pts = None
                if hasattr(lidar, "get_frame"):
                    try:
                        pts = lidar.get_frame()
                    except Exception as e:
                        print(f"get_frame error: {e}")
                        pts = None
                    pts = pts[~(pts[:, :5].sum(axis=1)==0)]
                    np.save(os.path.join(lidar_dir, f"points_{i:06d}.npy"), np.asarray(pts))
            i += 1
            # throttle by frame rate if given
            if args.fps and args.fps > 0:
                time.sleep(max(0, 1.0 / args.fps - (time.time() - t)))
    except KeyboardInterrupt:
        pass
    finally:
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
    p.add_argument("--fps", type=float, default=30.0, help="desired capture fps")
    p.add_argument("--host_ip", default="192.168.100.5",
                    help="CsdkLidar host ip")
    p.add_argument("--sdk_lib_path",
                    default="/home/rfor10/Livox-SDK2/build/sdk_core/liblivox_lidar_sdk_shared.so", help = "path to built Livox SDK2 shared library")
    p.add_argument("--config_path", default = "hap_config.json", help="config JSON file for SDK initialization")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args)