# lidarpy

Python project for interfacing with Livox HAP LiDAR hardware.

## Style
- Be extremely concise in all interactions and commit messages; sacrifice grammar for concision.

## Reference Materials
- Livox HAP communication protocol manual (primary parameterization source):
  https://github.com/Livox-SDK/Livox-SDK2/wiki/Livox-SDK-Communication-Protocol-HAP(English)
- Livox SDK2 source code: `Livox-SDK2-master/` (C++, CMake-based)
  - Headers: `Livox-SDK2-master/include/`
  - SDK core: `Livox-SDK2-master/sdk_core/`
  - Samples: `Livox-SDK2-master/samples/`
  - 3rd party: FastCRC, rapidjson, spdlog

## Project Structure
```
lidarpy/
├── __init__.py          # exports HapLidar, discover
├── constants.py         # enums: ports, cmd IDs, param keys, work modes, data types
├── protocol/
│   ├── packet.py        # CRC16/32, command build/parse, point packet parse
│   ├── commands.py      # key-value param encoding, command builders
│   └── discovery.py     # UDP broadcast device search
├── transport.py         # CommandChannel (send/recv), DataReceiver (background thread)
├── pointcloud.py        # point data → NumPy Nx5 array conversion
├── csdk.py              # ctypes wrapper for Livox SDK2 shared lib
├── lidar.py             # HapLidar: high-level sync/blocking API
└── webcam.py            # USB webcam image capture (V4L2/OpenCV)
tests/
Livox-SDK2-master/       # C++ SDK reference
```

## Commands
```bash
pip install -e ".[dev]"   # install with dev deps
pytest tests/ -v          # run tests
```

## Plans
- At the end of each plan, give a list of unresolved questions, if any.
  Questions should be extremely concise - sacrifice grammar for concision.
