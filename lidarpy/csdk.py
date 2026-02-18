"""ctypes wrapper for Livox SDK2 shared library (secondary backend).

Requires pre-built liblivox_lidar_sdk_shared.so/.dylib.
Build: cd Livox-SDK2-master && mkdir build && cd build && cmake .. && make
"""

import ctypes
import ctypes.util
import os
import sys
import time
import threading
import queue
from pathlib import Path

import numpy as np

from lidarpy.constants import WorkMode, PclDataType

# --- ctypes struct definitions (matching livox_lidar_def.h, pack=1) ---


class LivoxLidarSdkVer(ctypes.Structure):
    _fields_ = [("major", ctypes.c_int), ("minor", ctypes.c_int), ("patch", ctypes.c_int)]


class LivoxLidarInfo(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dev_type", ctypes.c_uint8),
        ("sn", ctypes.c_char * 16),
        ("lidar_ip", ctypes.c_char * 16),
    ]


class LivoxLidarEthernetPacket(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint8),
        ("length", ctypes.c_uint16),
        ("time_interval", ctypes.c_uint16),
        ("dot_num", ctypes.c_uint16),
        ("udp_cnt", ctypes.c_uint16),
        ("frame_cnt", ctypes.c_uint8),
        ("data_type", ctypes.c_uint8),
        ("time_type", ctypes.c_uint8),
        ("rsvd", ctypes.c_uint8 * 12),
        ("crc32", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint8 * 8),
        ("data", ctypes.c_uint8 * 1),  # variable-length
    ]


class LivoxLidarCartesianHighRawPoint(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_int32), ("y", ctypes.c_int32), ("z", ctypes.c_int32),
        ("reflectivity", ctypes.c_uint8), ("tag", ctypes.c_uint8),
    ]


class LivoxLidarCartesianLowRawPoint(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_int16), ("y", ctypes.c_int16), ("z", ctypes.c_int16),
        ("reflectivity", ctypes.c_uint8), ("tag", ctypes.c_uint8),
    ]


# --- Callback types ---

PointCloudCallbackType = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.c_uint8,
    ctypes.POINTER(LivoxLidarEthernetPacket), ctypes.c_void_p
)

InfoChangeCallbackType = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.POINTER(LivoxLidarInfo), ctypes.c_void_p
)

AsyncControlCallbackType = ctypes.CFUNCTYPE(
    None, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p
)


def _find_sdk_lib() -> str:
    """Locate the Livox SDK2 shared library."""
    candidates = []
    env_path = os.environ.get("LIVOX_SDK_LIB")
    if env_path:
        candidates.append(Path(env_path))
    base = Path(__file__).parent.parent / "Livox-SDK2-master" / "build"
    if sys.platform == "darwin":
        candidates.append(base / "sdk_core" / "liblivox_lidar_sdk_shared.dylib")
    else:
        candidates.append(base / "sdk_core" / "liblivox_lidar_sdk_shared.so")
    found = ctypes.util.find_library("livox_lidar_sdk_shared")
    if found:
        candidates.append(Path(found))

    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Livox SDK2 shared library not found. "
        "Build it: cd Livox-SDK2-master && mkdir build && cd build && cmake .. && make"
    )


class CsdkLidar:
    """Livox HAP interface via C SDK shared library (ctypes).

    Same high-level API as HapLidar but delegates to the C SDK.
    Requires config JSON file for SDK initialization.
    """

    def __init__(self, config_path: str, host_ip: str = ""):
        self._lib = ctypes.CDLL(_find_sdk_lib())
        self._config_path = config_path.encode()
        self._host_ip = host_ip.encode()
        self._point_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._handle: int | None = None
        self._device_info: LivoxLidarInfo | None = None
        self._discovered = threading.Event()
        self._initialized = False

        # Must keep references to prevent GC of ctypes callbacks
        self._pcl_cb = PointCloudCallbackType(self._on_point_cloud)
        self._info_cb = InfoChangeCallbackType(self._on_info_change)
        self._ctrl_cb = AsyncControlCallbackType(self._on_control_response)

        self._setup_api()

    def _setup_api(self):
        lib = self._lib
        lib.LivoxLidarSdkInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        lib.LivoxLidarSdkInit.restype = ctypes.c_bool
        lib.LivoxLidarSdkUninit.argtypes = []
        lib.LivoxLidarSdkUninit.restype = None
        lib.SetLivoxLidarPointCloudCallBack.argtypes = [PointCloudCallbackType, ctypes.c_void_p]
        lib.SetLivoxLidarPointCloudCallBack.restype = None
        lib.SetLivoxLidarInfoChangeCallback.argtypes = [InfoChangeCallbackType, ctypes.c_void_p]
        lib.SetLivoxLidarInfoChangeCallback.restype = None
        lib.SetLivoxLidarInfoCallback.argtypes = [
            ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.SetLivoxLidarInfoCallback.restype = None
        lib.SetLivoxLidarWorkMode.argtypes = [
            ctypes.c_uint32, ctypes.c_int, AsyncControlCallbackType, ctypes.c_void_p
        ]
        lib.SetLivoxLidarWorkMode.restype = ctypes.c_int32
        lib.DisableLivoxSdkConsoleLogger.argtypes = []
        lib.DisableLivoxSdkConsoleLogger.restype = None

    def connect(self, timeout: float = 10.0):
        """Init SDK and wait for device discovery."""
        # Register callbacks BEFORE init — SDK threads start inside Init
        self._lib.SetLivoxLidarPointCloudCallBack(self._pcl_cb, None)
        self._lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)
        self._lib.DisableLivoxSdkConsoleLogger()

        if not self._lib.LivoxLidarSdkInit(self._config_path, self._host_ip, None):
            raise RuntimeError(
                "LivoxLidarSdkInit failed — check config.json host_ip matches your NIC, "
                "and that ports 56000-59000 are not already in use"
            )
        self._initialized = True

        if not self._discovered.wait(timeout):
            raise TimeoutError("No device discovered within timeout")

    def start(self):
        """Set work mode to SAMPLING."""
        if self._handle is None:
            raise RuntimeError("No device connected")
        self._lib.SetLivoxLidarWorkMode(
            self._handle, WorkMode.SAMPLING, self._ctrl_cb, None
        )
        # Give SDK time to process mode change
        time.sleep(0.1)

    def stop(self):
        """Set work mode to IDLE."""
        if self._handle is not None:
            try:
                self._lib.SetLivoxLidarWorkMode(
                    self._handle, WorkMode.IDLE, self._ctrl_cb, None
                )
                time.sleep(0.1)
            except Exception:
                pass

    def get_frame(self, timeout: float = 1.0) -> np.ndarray:
        """Collect packets until udp_cnt resets or timeout. Returns Nx6 array.

        Columns: [x(m), y(m), z(m), reflectivity, tag, timestamp_us].
        """
        parts = []
        last_udp_cnt = -1
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                pkt = self._point_queue.get(timeout=0.05)
            except queue.Empty:
                if parts:
                    break
                continue
            udp_cnt, points = pkt
            if last_udp_cnt >= 0 and udp_cnt < last_udp_cnt and parts:
                break
            last_udp_cnt = udp_cnt
            if points.shape[0] > 0:
                parts.append(points)
        if not parts:
            return np.empty((0, 6), dtype=np.float64)
        return np.vstack(parts)

    def get_packet(self, timeout: float = 0.1) -> np.ndarray | None:
        """Return single packet's points."""
        try:
            _, points = self._point_queue.get(timeout=timeout)
            return points
        except queue.Empty:
            return None

    def close(self):
        """Only call Uninit if Init succeeded."""
        if self._initialized:
            try:
                self.stop()
                time.sleep(0.2)  # let SDK threads wind down
                self._lib.LivoxLidarSdkUninit()
            except Exception:
                pass
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- internal callbacks (called from C SDK threads) ---

    def _on_point_cloud(self, handle, dev_type, data_ptr, client_data):
        try:
            if not data_ptr:
                return
            pkt = data_ptr.contents
            dot_num = pkt.dot_num
            udp_cnt = pkt.udp_cnt
            if dot_num == 0:
                return
            data_type = pkt.data_type

            # Extract packet timestamp (8-byte LE nanoseconds) and per-point interval
            # time_interval is in units of 0.1 µs per SDK spec
            ts_bytes = bytes(pkt.timestamp)
            base_ns = int.from_bytes(ts_bytes, "little")
            dt_100ns = pkt.time_interval  # units of 0.1 µs (100 ns)

            raw_ptr = ctypes.cast(
                ctypes.addressof(pkt.data),
                ctypes.POINTER(ctypes.c_uint8)
            )
            if data_type == PclDataType.CARTESIAN_HIGH:
                arr_type = LivoxLidarCartesianHighRawPoint * dot_num
                raw = ctypes.cast(raw_ptr, ctypes.POINTER(arr_type)).contents
                points = np.empty((dot_num, 6), dtype=np.float64)
                for i in range(dot_num):
                    points[i, 0] = raw[i].x / 1000.0
                    points[i, 1] = raw[i].y / 1000.0
                    points[i, 2] = raw[i].z / 1000.0
                    points[i, 3] = raw[i].reflectivity
                    points[i, 4] = raw[i].tag
                    # timestamp in µs: base_ns/1000 + i * dt_100ns/10
                    points[i, 5] = base_ns / 1000.0 + i * (dt_100ns / 10.0)
            elif data_type == PclDataType.CARTESIAN_LOW:
                arr_type = LivoxLidarCartesianLowRawPoint * dot_num
                raw = ctypes.cast(raw_ptr, ctypes.POINTER(arr_type)).contents
                points = np.empty((dot_num, 6), dtype=np.float64)
                for i in range(dot_num):
                    points[i, 0] = raw[i].x / 100.0
                    points[i, 1] = raw[i].y / 100.0
                    points[i, 2] = raw[i].z / 100.0
                    points[i, 3] = raw[i].reflectivity
                    points[i, 4] = raw[i].tag
                    # timestamp in µs: base_ns/1000 + i * dt_100ns/10
                    points[i, 5] = base_ns / 1000.0 + i * (dt_100ns / 10.0)
            else:
                return
            self._point_queue.put_nowait((udp_cnt, points))
        except Exception:
            pass  # never let exceptions escape into C

    def _on_info_change(self, handle, info_ptr, client_data):
        try:
            if info_ptr:
                self._handle = handle
                self._device_info = info_ptr.contents
                self._discovered.set()
        except Exception:
            pass

    def _on_control_response(self, status, handle, response, client_data):
        pass
