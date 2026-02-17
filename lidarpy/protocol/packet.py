"""Livox HAP protocol packet build/parse with CRC."""

import struct
import binascii
from lidarpy.constants import (
    SOF, PROTOCOL_VERSION, HEADER_SIZE, POINT_HEADER_SIZE,
    CmdType, SenderType,
)

# CRC-16/CCITT-FALSE lookup table (poly 0x1021, init 0xFFFF)
_CRC16_TABLE = []
for _i in range(256):
    _crc = _i << 8
    for _ in range(8):
        _crc = ((_crc << 1) ^ 0x1021) if (_crc & 0x8000) else (_crc << 1)
    _CRC16_TABLE.append(_crc & 0xFFFF)


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc = ((_CRC16_TABLE[((crc >> 8) ^ b) & 0xFF]) ^ (crc << 8)) & 0xFFFF
    return crc


def crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def build_command(seq: int, cmd_id: int, data: bytes = b"",
                  sender_type: int = SenderType.HOST) -> bytes:
    """Build a Livox command frame (24-byte header + data)."""
    length = HEADER_SIZE + len(data)
    # Pack header up to CRC16 field (18 bytes)
    header_pre_crc = struct.pack(
        "<BBHIHBBxxxxxx",  # 6 reserved bytes via 'x' padding
        SOF, PROTOCOL_VERSION, length, seq, cmd_id,
        CmdType.REQ, sender_type,
    )  # 18 bytes
    crc16_val = crc16_ccitt(header_pre_crc)
    crc32_val = crc32(data) if data else 0
    header = header_pre_crc + struct.pack("<HI", crc16_val, crc32_val)
    return header + data


def parse_command_response(raw: bytes) -> dict:
    """Parse a command ACK packet. Returns dict with header fields + data."""
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"Packet too short: {len(raw)} < {HEADER_SIZE}")
    sof, ver, length, seq, cmd_id, cmd_type, sender_type = struct.unpack_from(
        "<BBHIHBBxxxxxx", raw
    )
    crc16_val, crc32_val = struct.unpack_from("<HI", raw, 18)
    data = raw[HEADER_SIZE:]
    # Verify CRC16
    expected_crc16 = crc16_ccitt(raw[:18])
    if crc16_val != expected_crc16:
        raise ValueError(f"CRC16 mismatch: got 0x{crc16_val:04X}, expected 0x{expected_crc16:04X}")
    # Verify CRC32 on data
    if data:
        expected_crc32 = crc32(data)
        if crc32_val != expected_crc32:
            raise ValueError(f"CRC32 mismatch: got 0x{crc32_val:08X}, expected 0x{expected_crc32:08X}")
    ret_code = data[0] if data else None
    return {
        "sof": sof,
        "version": ver,
        "length": length,
        "seq": seq,
        "cmd_id": cmd_id,
        "cmd_type": cmd_type,
        "sender_type": sender_type,
        "ret_code": ret_code,
        "data": data,
    }


def parse_point_packet(raw: bytes) -> dict:
    """Parse a point cloud / IMU data packet (36-byte header + points)."""
    if len(raw) < POINT_HEADER_SIZE:
        raise ValueError(f"Point packet too short: {len(raw)} < {POINT_HEADER_SIZE}")
    (version, length, time_interval, dot_num, udp_cnt,
     frame_cnt, data_type, time_type, pack_info) = struct.unpack_from(
        "<BHHHHBBBB", raw
    )
    # Skip 11 reserved bytes (offset 13-23)
    crc32_val = struct.unpack_from("<I", raw, 24)[0]
    timestamp = struct.unpack_from("<Q", raw, 28)[0]
    point_data = raw[POINT_HEADER_SIZE:]
    return {
        "version": version,
        "length": length,
        "time_interval": time_interval,
        "dot_num": dot_num,
        "udp_cnt": udp_cnt,
        "frame_cnt": frame_cnt,
        "data_type": data_type,
        "time_type": time_type,
        "pack_info": pack_info,
        "crc32": crc32_val,
        "timestamp": timestamp,
        "point_data": point_data,
    }
