"""UDP transport: command channel (send/recv) and data receiver (background thread)."""

import socket
import struct
import threading
import queue
from lidarpy.constants import Port, HEADER_SIZE
from lidarpy.protocol.packet import (
    build_command, parse_command_response, parse_point_packet,
)


class CommandChannel:
    """Send commands to LiDAR and block for ACK."""

    def __init__(self, lidar_ip: str, cmd_port: int = Port.HAP_CMD,
                 host_ip: str = "0.0.0.0", timeout: float = 1.0):
        self.lidar_ip = lidar_ip
        self.cmd_port = cmd_port
        self.timeout = timeout
        self._seq = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(timeout)
        self._sock.bind((host_ip, 0))

    @property
    def next_seq(self) -> int:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return seq

    def send_raw(self, data: bytes) -> dict:
        """Send raw packet bytes, wait for ACK, return parsed response."""
        self._sock.sendto(data, (self.lidar_ip, self.cmd_port))
        raw, _ = self._sock.recvfrom(2048)
        return parse_command_response(raw)

    def send_command(self, cmd_id: int, data: bytes = b"") -> dict:
        """Build and send command, return parsed ACK."""
        pkt = build_command(self.next_seq, cmd_id, data)
        return self.send_raw(pkt)

    def send_command_raw_packet(self, pkt: bytes) -> dict:
        """Send pre-built packet, return parsed ACK."""
        return self.send_raw(pkt)

    def close(self):
        self._sock.close()


class DataReceiver:
    """Background thread receiving UDP point/IMU packets into a queue."""

    def __init__(self, host_ip: str, port: int, buf_size: int = 4096,
                 max_queue: int = 1000):
        self.host_ip = host_ip
        self.port = port
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self._sock.bind((host_ip, port))
        self._sock.settimeout(0.5)
        self._buf_size = buf_size
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _recv_loop(self):
        while self._running:
            try:
                raw, addr = self._sock.recvfrom(self._buf_size)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                parsed = parse_point_packet(raw)
                self.queue.put(parsed, block=False)
            except (ValueError, queue.Full):
                continue

    def get(self, timeout: float = 0.1) -> dict | None:
        """Get next parsed packet from queue, or None on timeout."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.stop()
        self._sock.close()
