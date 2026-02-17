from enum import IntEnum

# Protocol framing
SOF = 0xAA
PROTOCOL_VERSION = 0x00
HEADER_SIZE = 24  # bytes before data payload
MAX_FRAME_SIZE = 1400

# Point data packet header size (before point array)
POINT_HEADER_SIZE = 36


# --- Ports ---
class Port(IntEnum):
    HAP_CMD = 56000
    HAP_POINT = 57000
    HAP_IMU = 58000
    HAP_LOG = 59000
    DETECTION_LISTEN = 56001


# --- Command IDs ---
class CmdId(IntEnum):
    SEARCH = 0x0000
    SET_PARAM = 0x0100
    QUERY_PARAM = 0x0101
    PUSH_MSG = 0x0102
    REBOOT = 0x0200
    RESET = 0x0201
    PUSH_LOG = 0x0300
    COLLECTION_LOG = 0x0301
    DEBUG_POINT_CLOUD = 0x0303
    UPGRADE_START = 0x0400
    UPGRADE_DATA = 0x0401
    UPGRADE_COMPLETE = 0x0402
    UPGRADE_STATUS = 0x0403
    FW_INFO = 0x00FF


# --- Command types ---
class CmdType(IntEnum):
    REQ = 0x00
    ACK = 0x01


class SenderType(IntEnum):
    HOST = 0x00
    LIDAR = 0x01


# --- Parameter keys (for SET_PARAM / QUERY_PARAM) ---
class ParamKey(IntEnum):
    PCL_DATA_TYPE = 0x0000
    SCAN_PATTERN = 0x0001
    DUAL_EMIT = 0x0002
    POINT_SEND_EN = 0x0003
    LIDAR_IP = 0x0004
    STATE_INFO_HOST_IP = 0x0005
    POINT_DATA_HOST_IP = 0x0006
    IMU_DATA_HOST_IP = 0x0007
    LOG_HOST_IP = 0x0009
    INSTALL_ATTITUDE = 0x0012
    BLIND_SPOT = 0x0013
    WORK_MODE = 0x001A
    GLASS_HEAT = 0x001B
    IMU_DATA_EN = 0x001C
    FUSA_EN = 0x001D
    FORCED_HEATING = 0x001E
    WORK_MODE_AFTER_BOOT = 0x0020
    # Read-only keys (query only)
    SN = 0x8000
    PRODUCT_INFO = 0x8001
    FW_VERSION = 0x8002
    LOADER_VERSION = 0x8003
    HW_VERSION = 0x8004
    MAC = 0x8005
    CUR_WORK_STATE = 0x8006
    STATUS_CODE = 0x800D
    DIAG_STATUS = 0x800E
    FLASH_STATUS = 0x800F
    FW_TYPE = 0x8010
    GLASS_HEAT_STATE = 0x8012


# --- Work modes ---
class WorkMode(IntEnum):
    SAMPLING = 0x01
    IDLE = 0x02
    SLEEP = 0x03
    ERROR = 0x04
    SELFCHECK = 0x05
    MOTOR_STARTUP = 0x06
    MOTOR_STOP = 0x07
    UPGRADE = 0x08


# --- Point cloud data types ---
class PclDataType(IntEnum):
    IMU = 0x00
    CARTESIAN_HIGH = 0x01
    CARTESIAN_LOW = 0x02
    SPHERICAL = 0x03


# --- Scan patterns ---
class ScanPattern(IntEnum):
    NON_REPETITIVE = 0x00
    REPETITIVE = 0x01


# --- Device types ---
class DeviceType(IntEnum):
    HUB = 0
    MID40 = 1
    TELE = 2
    HORIZON = 3
    MID70 = 6
    AVIA = 7
    MID360 = 9
    HAP = 10


# --- Return codes ---
class RetCode(IntEnum):
    SUCCESS = 0x00
    FAILURE = 0x01
    NOT_PERMIT = 0x02
    OUT_OF_RANGE = 0x03
    PARAM_NOT_SUPPORT = 0x20
    PARAM_REBOOT_EFFECT = 0x21
    PARAM_READ_ONLY = 0x22
    PARAM_INVALID_LEN = 0x23
    PARAM_KEY_NUM_ERR = 0x24


# --- Point struct sizes (bytes per point) ---
CARTESIAN_HIGH_POINT_SIZE = 14  # int32 x,y,z + u8 refl + u8 tag
CARTESIAN_LOW_POINT_SIZE = 8   # int16 x,y,z + u8 refl + u8 tag
SPHERICAL_POINT_SIZE = 10      # u32 depth + u16 theta + u16 phi + u8 refl + u8 tag
IMU_POINT_SIZE = 24            # 6x float32
