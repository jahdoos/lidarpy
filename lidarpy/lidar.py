"""High-level sync API for Livox HAP LiDAR."""

import struct
import time
import numpy as np
from lidarpy.constants import (
    Port, ParamKey, WorkMode, PclDataType, ScanPattern, RetCode,
)
from lidarpy.protocol.discovery import DeviceInfo, discover, _parse_detection_data
from lidarpy.protocol.commands import (
    build_search, build_set_work_mode, build_set_params,
    build_query_params, build_set_pcl_data_type, build_enable_point_send,
    build_set_imu, build_reboot, decode_key_values,
)
from lidarpy.protocol.packet import parse_command_response
from lidarpy.transport import CommandChannel, DataReceiver
from lidarpy.pointcloud import decode_points, decode_imu


class HapLidar:
    """Blocking interface to a single Livox HAP LiDAR.

    Usage:
        with HapLidar(host_ip="192.168.1.50") as lidar:
            lidar.connect()
            lidar.start()
            frame = lidar.get_frame()
            lidar.stop()
    """

    def __init__(self, host_ip: str, ip: str = "192.168.1.100",
                 cmd_port: int = Port.HAP_CMD,
                 point_port: int = Port.HAP_POINT):
        self.host_ip = host_ip
        self.ip = ip
        self.cmd_port = cmd_port
        self.point_port = point_port
        self._cmd: CommandChannel | None = None
        self._data: DataReceiver | None = None
        self._device_info: DeviceInfo | None = None

    # --- lifecycle ---

    def connect(self) -> DeviceInfo:
        """Open command channel, send search to verify device, return info."""
        self._cmd = CommandChannel(self.ip, self.cmd_port, self.host_ip)
        pkt = build_search(self._cmd.next_seq)
        resp = self._cmd.send_raw(pkt)
        self._device_info = _parse_detection_data(resp["data"])
        return self._device_info

    def start(self):
        """Start point cloud streaming: set SAMPLING mode, open data receiver."""
        self._ensure_connected()
        pkt = build_set_work_mode(self._cmd.next_seq, WorkMode.SAMPLING)
        resp = self._cmd.send_raw(pkt)
        self._data = DataReceiver(self.host_ip, self.point_port)
        self._data.start()

    def stop(self):
        """Stop streaming: set IDLE mode, close data receiver."""
        if self._data:
            self._data.stop()
            self._data.close()
            self._data = None
        if self._cmd:
            try:
                pkt = build_set_work_mode(self._cmd.next_seq, WorkMode.IDLE)
                self._cmd.send_raw(pkt)
            except Exception:
                pass

    def close(self):
        """Cleanup all resources."""
        self.stop()
        if self._cmd:
            self._cmd.close()
            self._cmd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- data capture ---

    def get_frame(self, timeout: float = 1.0) -> np.ndarray:
        """Block until full frame (udp_cnt resets) or timeout. Returns Nx5 array."""
        self._ensure_streaming()
        parts = []
        last_udp_cnt = -1
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            pkt = self._data.get(timeout=min(remaining, 0.05))
            if pkt is None:
                if parts:
                    break  # timeout fallback: flush what we have
                continue
            udp_cnt = pkt["udp_cnt"]
            # Detect frame boundary: udp_cnt wrapped back to 0
            if last_udp_cnt >= 0 and udp_cnt < last_udp_cnt and parts:
                # Put this packet back conceptually — but simpler to include it
                # in next frame. For now, flush current frame.
                # Re-queue this packet for next get_frame call.
                try:
                    self._data.queue.put(pkt, block=False)
                except Exception:
                    pass
                break
            last_udp_cnt = udp_cnt
            try:
                pts = decode_points(pkt)
                if pts.shape[0] > 0:
                    parts.append(pts)
            except ValueError:
                continue

        if not parts:
            return np.empty((0, 5), dtype=np.float64)
        return np.vstack(parts)

    def get_packet(self, timeout: float = 0.1) -> np.ndarray | None:
        """Return single packet's points as Nx5 array, or None on timeout."""
        self._ensure_streaming()
        pkt = self._data.get(timeout=timeout)
        if pkt is None:
            return None
        return decode_points(pkt)

    # --- configuration ---

    def configure(self, **kwargs):
        """Set device params. Supported kwargs:
            pcl_data_type: PclDataType enum value
            scan_pattern: ScanPattern enum value
            blind_spot: int (cm, 50-200)
            imu: bool
            point_send: bool
        """
        self._ensure_connected()
        params = {}
        if "pcl_data_type" in kwargs:
            params[ParamKey.PCL_DATA_TYPE] = struct.pack("<B", kwargs["pcl_data_type"])
        if "scan_pattern" in kwargs:
            params[ParamKey.SCAN_PATTERN] = struct.pack("<B", kwargs["scan_pattern"])
        if "blind_spot" in kwargs:
            params[ParamKey.BLIND_SPOT] = struct.pack("<I", kwargs["blind_spot"])
        if "imu" in kwargs:
            params[ParamKey.IMU_DATA_EN] = struct.pack("<B", int(kwargs["imu"]))
        if "point_send" in kwargs:
            val = 0x00 if kwargs["point_send"] else 0x01
            params[ParamKey.POINT_SEND_EN] = struct.pack("<B", val)
        if params:
            pkt = build_set_params(self._cmd.next_seq, params)
            self._cmd.send_raw(pkt)

    def query(self, *keys: int) -> dict[int, bytes]:
        """Query device parameters by key. Returns {key: raw_value_bytes}."""
        self._ensure_connected()
        pkt = build_query_params(self._cmd.next_seq, list(keys))
        resp = self._cmd.send_raw(pkt)
        return decode_key_values(resp["data"])

    def reboot(self, delay_ms: int = 100):
        """Reboot device after delay_ms (100-2000)."""
        self._ensure_connected()
        pkt = build_reboot(self._cmd.next_seq, delay_ms)
        self._cmd.send_raw(pkt)

    # --- properties ---

    @property
    def device_info(self) -> DeviceInfo | None:
        return self._device_info

    # --- internal ---

    def _ensure_connected(self):
        if self._cmd is None:
            raise RuntimeError("Not connected. Call connect() first.")

    def _ensure_streaming(self):
        if self._data is None:
            raise RuntimeError("Not streaming. Call start() first.")
