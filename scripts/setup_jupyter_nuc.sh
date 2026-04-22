#!/usr/bin/env bash
# Install JupyterLab service for lidarpy on Ubuntu, bind 0.0.0.0:9999, password
# auth, systemd on boot. Uses existing conda env (jupyterlab preinstalled).
# Run as the user that should own the server (not root):
#   cd /path/to/lidarpy && bash scripts/setup_jupyter_nuc.sh
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
    echo "Run as your normal user, not root. sudo is invoked where needed." >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-$HOME/miniconda3/envs/lidar}"
JUPYTER="${CONDA_ENV}/bin/jupyter"
PORT=9999
SERVICE_NAME=jupyter-lidarpy
RUN_USER="$USER"
CONFIG_DIR="${HOME}/.jupyter"
CONFIG_FILE="${CONFIG_DIR}/jupyter_lab_config.py"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ ! -x "$JUPYTER" ]; then
    echo "jupyter not found at $JUPYTER" >&2
    exit 1
fi

echo "repo=$REPO_DIR user=$RUN_USER env=$CONDA_ENV port=$PORT"

# password (writes hash to ~/.jupyter/jupyter_server_config.json)
mkdir -p "$CONFIG_DIR"
echo
echo "Set a Jupyter password (entered twice):"
"$JUPYTER" lab password

# server config
cat > "$CONFIG_FILE" <<EOF
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = ${PORT}
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.root_dir = r'${REPO_DIR}'
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_origin = '*'
EOF

# systemd unit — PATH includes conda env so kernels/subprocesses resolve
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=JupyterLab for lidarpy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
Environment=PATH=${CONDA_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=${JUPYTER} lab --config=${CONFIG_FILE}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

if command -v ufw >/dev/null 2>&1 && sudo ufw status | grep -q "Status: active"; then
    sudo ufw allow ${PORT}/tcp
fi

echo
echo "Up at http://192.168.1.200:${PORT}"
echo "status:  sudo systemctl status ${SERVICE_NAME}"
echo "logs:    sudo journalctl -u ${SERVICE_NAME} -f"
