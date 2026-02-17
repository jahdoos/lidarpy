"""Livox HAP command builders and key-value param encoding."""

import struct
from lidarpy.constants import CmdId, ParamKey, WorkMode
from lidarpy.protocol.packet import build_command


def encode_key_value(key: int, value: bytes) -> bytes:
    """Encode single key-value pair: key(u16) + length(u16) + value."""
    return struct.pack("<HH", key, len(value)) + value


def encode_key_values(params: dict[int, bytes]) -> bytes:
    """Encode SET_PARAM payload: key_num(u16) + rsvd(u16) + key_value_list."""
    kvs = b"".join(encode_key_value(k, v) for k, v in params.items())
    return struct.pack("<HH", len(params), 0) + kvs


def encode_query_keys(keys: list[int]) -> bytes:
    """Encode QUERY_PARAM payload: key_num(u16) + rsvd(u16) + keys."""
    return struct.pack("<HH", len(keys), 0) + b"".join(
        struct.pack("<H", k) for k in keys
    )


def decode_key_values(data: bytes) -> dict[int, bytes]:
    """Decode key-value list from QUERY_PARAM response (after ret_code)."""
    if len(data) < 5:
        return {}
    ret_code = data[0]
    key_num = struct.unpack_from("<H", data, 1)[0]
    # rsvd at offset 3
    result = {}
    offset = 5
    for _ in range(key_num):
        if offset + 4 > len(data):
            break
        key, length = struct.unpack_from("<HH", data, offset)
        offset += 4
        result[key] = data[offset:offset + length]
        offset += length
    return result


def build_search(seq: int) -> bytes:
    """Build device search command (0x0000, empty payload)."""
    return build_command(seq, CmdId.SEARCH)


def build_set_params(seq: int, params: dict[int, bytes]) -> bytes:
    """Build SET_PARAM command."""
    return build_command(seq, CmdId.SET_PARAM, encode_key_values(params))


def build_query_params(seq: int, keys: list[int]) -> bytes:
    """Build QUERY_PARAM command."""
    return build_command(seq, CmdId.QUERY_PARAM, encode_query_keys(keys))


def build_set_work_mode(seq: int, mode: int) -> bytes:
    """Set work mode via SET_PARAM with key 0x001A."""
    return build_set_params(seq, {ParamKey.WORK_MODE: struct.pack("<B", mode)})


def build_reboot(seq: int, delay_ms: int = 100) -> bytes:
    """Build reboot command. delay_ms: 100-2000."""
    return build_command(seq, CmdId.REBOOT, struct.pack("<H", delay_ms))


def build_set_pcl_data_type(seq: int, data_type: int) -> bytes:
    """Set point cloud data type (Cartesian high/low, spherical)."""
    return build_set_params(seq, {ParamKey.PCL_DATA_TYPE: struct.pack("<B", data_type)})


def build_enable_point_send(seq: int, enable: bool = True) -> bytes:
    """Enable/disable point cloud transmission. 0x00=enable, 0x01=disable."""
    val = 0x00 if enable else 0x01
    return build_set_params(seq, {ParamKey.POINT_SEND_EN: struct.pack("<B", val)})


def build_set_imu(seq: int, enable: bool = True) -> bytes:
    """Enable/disable IMU data."""
    return build_set_params(seq, {ParamKey.IMU_DATA_EN: struct.pack("<B", int(enable))})
