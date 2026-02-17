"""Point cloud / IMU data → NumPy array conversion."""

import struct
import numpy as np
from lidarpy.constants import (
    PclDataType, CARTESIAN_HIGH_POINT_SIZE, CARTESIAN_LOW_POINT_SIZE,
    SPHERICAL_POINT_SIZE, IMU_POINT_SIZE,
)


def decode_points(packet: dict) -> np.ndarray:
    """Convert parsed point packet to Nx5 float64 array [x(m), y(m), z(m), reflectivity, tag].

    Handles Cartesian high, Cartesian low, and spherical formats.
    """
    data = packet["point_data"]
    dot_num = packet["dot_num"]
    data_type = packet["data_type"]

    if dot_num == 0:
        return np.empty((0, 5), dtype=np.float64)

    if data_type == PclDataType.CARTESIAN_HIGH:
        return _decode_cartesian_high(data, dot_num)
    elif data_type == PclDataType.CARTESIAN_LOW:
        return _decode_cartesian_low(data, dot_num)
    elif data_type == PclDataType.SPHERICAL:
        return _decode_spherical(data, dot_num)
    else:
        raise ValueError(f"Unknown point data type: {data_type}")


def _decode_cartesian_high(data: bytes, n: int) -> np.ndarray:
    """int32 x,y,z (mm) + u8 refl + u8 tag → float64 [x(m),y(m),z(m),refl,tag]."""
    dt = np.dtype([
        ("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
        ("refl", "u1"), ("tag", "u1"),
    ])
    raw = np.frombuffer(data, dtype=dt, count=n)
    out = np.empty((n, 5), dtype=np.float64)
    out[:, 0] = raw["x"] / 1000.0
    out[:, 1] = raw["y"] / 1000.0
    out[:, 2] = raw["z"] / 1000.0
    out[:, 3] = raw["refl"]
    out[:, 4] = raw["tag"]
    return out


def _decode_cartesian_low(data: bytes, n: int) -> np.ndarray:
    """int16 x,y,z (10mm units) + u8 refl + u8 tag → float64 [x(m),y(m),z(m),refl,tag]."""
    dt = np.dtype([
        ("x", "<i2"), ("y", "<i2"), ("z", "<i2"),
        ("refl", "u1"), ("tag", "u1"),
    ])
    raw = np.frombuffer(data, dtype=dt, count=n)
    out = np.empty((n, 5), dtype=np.float64)
    out[:, 0] = raw["x"] / 100.0
    out[:, 1] = raw["y"] / 100.0
    out[:, 2] = raw["z"] / 100.0
    out[:, 3] = raw["refl"]
    out[:, 4] = raw["tag"]
    return out


def _decode_spherical(data: bytes, n: int) -> np.ndarray:
    """u32 depth(mm) + u16 theta + u16 phi + u8 refl + u8 tag → float64 [x,y,z,refl,tag]."""
    dt = np.dtype([
        ("depth", "<u4"), ("theta", "<u2"), ("phi", "<u2"),
        ("refl", "u1"), ("tag", "u1"),
    ])
    raw = np.frombuffer(data, dtype=dt, count=n)
    depth_m = raw["depth"] / 1000.0
    theta = raw["theta"] / 100.0 * (np.pi / 180.0)  # centidegrees → radians
    phi = raw["phi"] / 100.0 * (np.pi / 180.0)
    out = np.empty((n, 5), dtype=np.float64)
    out[:, 0] = depth_m * np.cos(phi) * np.cos(theta)
    out[:, 1] = depth_m * np.cos(phi) * np.sin(theta)
    out[:, 2] = depth_m * np.sin(phi)
    out[:, 3] = raw["refl"]
    out[:, 4] = raw["tag"]
    return out


def decode_imu(packet: dict) -> np.ndarray:
    """Convert IMU packet to Nx6 float64 array [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]."""
    data = packet["point_data"]
    dot_num = packet["dot_num"]
    if dot_num == 0:
        return np.empty((0, 6), dtype=np.float64)
    dt = np.dtype([
        ("gx", "<f4"), ("gy", "<f4"), ("gz", "<f4"),
        ("ax", "<f4"), ("ay", "<f4"), ("az", "<f4"),
    ])
    raw = np.frombuffer(data, dtype=dt, count=dot_num)
    out = np.empty((dot_num, 6), dtype=np.float64)
    out[:, 0] = raw["gx"]
    out[:, 1] = raw["gy"]
    out[:, 2] = raw["gz"]
    out[:, 3] = raw["ax"]
    out[:, 4] = raw["ay"]
    out[:, 5] = raw["az"]
    return out
