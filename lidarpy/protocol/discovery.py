"""Livox HAP device discovery via UDP broadcast."""

import socket
import struct
import time
from dataclasses import dataclass
from lidarpy.constants import Port, DeviceType
from lidarpy.protocol.commands import build_search
from lidarpy.protocol.packet import parse_command_response


@dataclass
class DeviceInfo:
    dev_type: int
    sn: str
    ip: str
    cmd_port: int

    @property
    def dev_type_name(self) -> str:
        try:
            return DeviceType(self.dev_type).name
        except ValueError:
            return f"UNKNOWN({self.dev_type})"


def _parse_detection_data(data: bytes) -> DeviceInfo:
    """Parse detection response payload: ret(1)+dev_type(1)+SN(16)+IP(4)+port(2)."""
    if len(data) < 24:
        raise ValueError(f"Detection data too short: {len(data)}")
    ret_code = data[0]
    dev_type = data[1]
    sn = data[2:18].rstrip(b"\x00").decode("ascii", errors="replace")
    ip_bytes = data[18:22]
    ip = f"{ip_bytes[0]}.{ip_bytes[1]}.{ip_bytes[2]}.{ip_bytes[3]}"
    cmd_port = struct.unpack_from("<H", data, 22)[0]
    return DeviceInfo(dev_type=dev_type, sn=sn, ip=ip, cmd_port=cmd_port)


def discover(timeout: float = 3.0, interface: str = "0.0.0.0") -> list[DeviceInfo]:
    """Broadcast search and collect responding Livox devices.

    Args:
        timeout: seconds to listen for responses
        interface: local IP to bind (use NIC IP on the LiDAR subnet)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    sock.bind((interface, 0))

    pkt = build_search(seq=0)
    sock.sendto(pkt, ("255.255.255.255", Port.HAP_CMD))

    devices = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        try:
            resp = parse_command_response(raw)
            if resp["cmd_type"] == 0x01 and resp["data"]:
                info = _parse_detection_data(resp["data"])
                if not any(d.sn == info.sn for d in devices):
                    devices.append(info)
        except (ValueError, IndexError):
            continue

    sock.close()
    return devices
