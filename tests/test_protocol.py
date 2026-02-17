"""Tests for protocol layer: CRC, packet build/parse, commands, pointcloud."""

import struct
import binascii
import numpy as np
import pytest

from lidarpy.constants import (
    SOF, HEADER_SIZE, POINT_HEADER_SIZE, CmdId, CmdType, SenderType,
    ParamKey, WorkMode, PclDataType,
    CARTESIAN_HIGH_POINT_SIZE, CARTESIAN_LOW_POINT_SIZE,
)
from lidarpy.protocol.packet import (
    crc16_ccitt, crc32, build_command, parse_command_response, parse_point_packet,
)
from lidarpy.protocol.commands import (
    encode_key_values, encode_key_value, encode_query_keys, decode_key_values,
    build_search, build_set_work_mode, build_set_pcl_data_type,
    build_enable_point_send, build_reboot,
)
from lidarpy.pointcloud import decode_points, decode_imu


# --- CRC tests ---

class TestCRC:
    def test_crc32_empty(self):
        assert crc32(b"") == 0x00000000

    def test_crc32_matches_binascii(self):
        data = b"hello world"
        assert crc32(data) == (binascii.crc32(data) & 0xFFFFFFFF)

    def test_crc16_known_value(self):
        # CRC-16/CCITT-FALSE of "123456789" = 0x29B1
        result = crc16_ccitt(b"123456789")
        assert result == 0x29B1

    def test_crc16_empty(self):
        assert crc16_ccitt(b"") == 0xFFFF


# --- Packet build/parse round-trip ---

class TestPacket:
    def test_build_command_header_size(self):
        pkt = build_command(0, CmdId.SEARCH)
        assert len(pkt) == HEADER_SIZE

    def test_build_command_with_data(self):
        data = b"\x01\x02\x03"
        pkt = build_command(42, CmdId.SET_PARAM, data)
        assert len(pkt) == HEADER_SIZE + 3
        assert pkt[0] == SOF

    def test_round_trip_no_data(self):
        pkt = build_command(7, CmdId.SEARCH)
        parsed = parse_command_response(pkt)
        assert parsed["sof"] == SOF
        assert parsed["seq"] == 7
        assert parsed["cmd_id"] == CmdId.SEARCH
        assert parsed["cmd_type"] == CmdType.REQ

    def test_round_trip_with_data(self):
        data = b"\xAB\xCD"
        pkt = build_command(100, CmdId.REBOOT, data)
        parsed = parse_command_response(pkt)
        assert parsed["seq"] == 100
        assert parsed["cmd_id"] == CmdId.REBOOT
        assert parsed["data"] == data

    def test_crc16_verification_fails_on_corrupt(self):
        pkt = bytearray(build_command(0, CmdId.SEARCH))
        pkt[5] ^= 0xFF  # corrupt a header byte
        with pytest.raises(ValueError, match="CRC16"):
            parse_command_response(bytes(pkt))

    def test_crc32_verification_fails_on_corrupt(self):
        pkt = bytearray(build_command(0, CmdId.SET_PARAM, b"\x01\x02"))
        pkt[-1] ^= 0xFF  # corrupt data byte
        with pytest.raises(ValueError, match="CRC32"):
            parse_command_response(bytes(pkt))

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            parse_command_response(b"\x00" * 10)


# --- Point packet parsing ---

class TestPointPacket:
    def _make_point_packet(self, dot_num, data_type, point_data):
        """Build a synthetic point data UDP packet."""
        # 36-byte header
        hdr = struct.pack(
            "<BHHHHBBBB",
            0x00,  # version
            POINT_HEADER_SIZE + len(point_data),  # length
            100,   # time_interval
            dot_num,
            0,     # udp_cnt
            0,     # frame_cnt
            data_type,
            0,     # time_type
            0,     # pack_info
        )
        hdr += b"\x00" * 11  # reserved
        hdr += struct.pack("<I", crc32(point_data))  # crc32
        hdr += struct.pack("<Q", 1000000)  # timestamp
        return hdr + point_data

    def test_parse_cartesian_high(self):
        # 1 point: x=1000mm, y=2000mm, z=3000mm, refl=128, tag=1
        pt = struct.pack("<iiiBB", 1000, 2000, 3000, 128, 1)
        raw = self._make_point_packet(1, PclDataType.CARTESIAN_HIGH, pt)
        parsed = parse_point_packet(raw)
        assert parsed["dot_num"] == 1
        assert parsed["data_type"] == PclDataType.CARTESIAN_HIGH

    def test_parse_empty_packet(self):
        raw = self._make_point_packet(0, PclDataType.CARTESIAN_HIGH, b"")
        parsed = parse_point_packet(raw)
        assert parsed["dot_num"] == 0


# --- Commands ---

class TestCommands:
    def test_encode_key_value(self):
        kv = encode_key_value(0x001A, b"\x01")
        assert kv == struct.pack("<HH", 0x001A, 1) + b"\x01"

    def test_encode_key_values_payload(self):
        payload = encode_key_values({0x001A: b"\x01"})
        key_num, rsvd = struct.unpack_from("<HH", payload)
        assert key_num == 1
        assert rsvd == 0

    def test_encode_query_keys(self):
        payload = encode_query_keys([0x8000, 0x8001])
        key_num = struct.unpack_from("<H", payload)[0]
        assert key_num == 2

    def test_decode_key_values_round_trip(self):
        # Simulate a response: ret_code(1) + key_num(2) + rsvd(2) + key_value_list
        kv_data = encode_key_value(0x8000, b"SN123456\x00\x00\x00\x00\x00\x00\x00\x00")
        resp = struct.pack("<BHH", 0x00, 1, 0) + kv_data
        result = decode_key_values(resp)
        assert 0x8000 in result
        assert result[0x8000][:8] == b"SN123456"

    def test_build_search(self):
        pkt = build_search(0)
        assert len(pkt) == HEADER_SIZE
        parsed = parse_command_response(pkt)
        assert parsed["cmd_id"] == CmdId.SEARCH

    def test_build_set_work_mode(self):
        pkt = build_set_work_mode(1, WorkMode.SAMPLING)
        parsed = parse_command_response(pkt)
        assert parsed["cmd_id"] == CmdId.SET_PARAM

    def test_build_reboot(self):
        pkt = build_reboot(5, delay_ms=200)
        parsed = parse_command_response(pkt)
        assert parsed["cmd_id"] == CmdId.REBOOT
        delay = struct.unpack_from("<H", parsed["data"])[0]
        assert delay == 200


# --- Point cloud decode ---

class TestPointCloudDecode:
    def _make_parsed_packet(self, dot_num, data_type, point_data):
        return {
            "dot_num": dot_num,
            "data_type": data_type,
            "point_data": point_data,
        }

    def test_cartesian_high_decode(self):
        # 2 points
        pts = struct.pack("<iiiBB", 1000, 2000, 3000, 200, 0)
        pts += struct.pack("<iiiBB", -500, 100, 4500, 50, 1)
        pkt = self._make_parsed_packet(2, PclDataType.CARTESIAN_HIGH, pts)
        arr = decode_points(pkt)
        assert arr.shape == (2, 5)
        np.testing.assert_allclose(arr[0, :3], [1.0, 2.0, 3.0])
        assert arr[0, 3] == 200
        assert arr[0, 4] == 0
        np.testing.assert_allclose(arr[1, :3], [-0.5, 0.1, 4.5])

    def test_cartesian_low_decode(self):
        pts = struct.pack("<hhhBB", 100, 200, 300, 128, 0)
        pkt = self._make_parsed_packet(1, PclDataType.CARTESIAN_LOW, pts)
        arr = decode_points(pkt)
        assert arr.shape == (1, 5)
        np.testing.assert_allclose(arr[0, :3], [1.0, 2.0, 3.0])

    def test_empty_returns_empty_array(self):
        pkt = self._make_parsed_packet(0, PclDataType.CARTESIAN_HIGH, b"")
        arr = decode_points(pkt)
        assert arr.shape == (0, 5)

    def test_imu_decode(self):
        imu = struct.pack("<ffffff", 0.1, 0.2, 0.3, 9.8, 0.0, -0.1)
        pkt = {"dot_num": 1, "data_type": PclDataType.IMU, "point_data": imu}
        arr = decode_imu(pkt)
        assert arr.shape == (1, 6)
        np.testing.assert_allclose(arr[0], [0.1, 0.2, 0.3, 9.8, 0.0, -0.1], atol=1e-6)

    def test_spherical_decode(self):
        # depth=5000mm, theta=9000 centideg (90°), phi=0
        pts = struct.pack("<IHHBB", 5000, 9000, 0, 100, 0)
        pkt = self._make_parsed_packet(1, PclDataType.SPHERICAL, pts)
        arr = decode_points(pkt)
        assert arr.shape == (1, 5)
        # At theta=90°, phi=0: x≈0, y≈5.0, z≈0
        np.testing.assert_allclose(arr[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(arr[0, 1], 5.0, atol=1e-6)
        np.testing.assert_allclose(arr[0, 2], 0.0, atol=1e-6)
