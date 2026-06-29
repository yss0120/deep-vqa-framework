#!/usr/bin/env bash
# --- optimize_env.sh ---

# Tasks:
# 1. release ports
# 2. change network cards' mtu
# 3. check proxy; set http/https config if needed
# 4. Websocket control
#   - iopub_data_rate_limit
#   - rate_limit_window
#   - terminado_settings: inactive_timeout, ping_interval
#   - tornado_settings
#   - websocket_ping_interval
#   - websocket_ping_timeout
#   - iopub_msg_rate_limit
#   - allow_origin
#   - allow_remote_access
#   - disable_check_xsrf
# 5. Terminal silencing and flow control
# 6. Kernel idle cleanup
#   - cull_idle_timeout
#   - cull_connected
#   - cull_busy


# Notes:
# YAML/JSON separated "data-driven" programming is terrible in that case.
set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}=== Network & Terminal Optimization ===${NC}"
echo -e "${BLUE}========================================${NC}"

# Helper: sudo handling
HAS_SUDO=false
if command -v sudo &> /dev/null; then
    HAS_SUDO=true
fi

if [ "$(id -u)" -eq 0 ]; then
    ADMIN_SUDO=""
else
    if [ "$HAS_SUDO" = true ]; then
        ADMIN_SUDO="sudo"
    else
        ADMIN_SUDO=""
        echo -e "${YELLOW}[!] No sudo available, some operations may fail${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# 1. Release ports (kill processes occupying common ports)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Releasing common ports...${NC}"
COMMON_PORTS=(6006 8888 8080 8000 7860)
for port in "${COMMON_PORTS[@]}"; do
    # Find PID using the port
    PID=$(ss -tlnp 2>/dev/null | grep -E ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -n "$PID" ]; then
        echo -e "  Port $port: killing process $PID"
        kill -9 "$PID" 2>/dev/null || true
    else
        echo -e "  Port $port: free"
    fi
done

# -----------------------------------------------------------------------------
# 2. Change network card's MTU
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Adjusting MTU settings...${NC}"
# Try common interface names: eth0, ens3, ens5, eno1
for iface in eth0 ens3 ens5 eno1; do
    if ip link show "$iface" &>/dev/null; then
        ${ADMIN_SUDO} ip link set dev "$iface" mtu 1400 2>/dev/null && \
            echo -e "  ${GREEN}[√] Set $iface MTU to 1400${NC}" || \
            echo -e "  ${YELLOW}[!] Failed to set MTU for $iface${NC}"
        break
    fi
done

# -----------------------------------------------------------------------------
# 3. Check proxy; set http/https config if needed
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Checking proxy configuration...${NC}"
if curl -s -I --max-time 3 "https://www.google.com" -o /dev/null -w "%{http_code}" | grep -qE "200|301|302"; then
    echo -e "  ${GREEN}[√] Direct internet connection works, no proxy needed${NC}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
else
    echo -e "  ${YELLOW}[!] Direct connection failed, attempting to set proxy...${NC}"
    # Common proxy ports: 7890 (Clash), 10809 (v2ray), 8118 (Tor)
    for proxy_port in 7890 7897 10809 8118; do
        if curl -s -I --max-time 2 --proxy "http://127.0.0.1:$proxy_port" \
            "https://www.google.com" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|301|302"; then
            export http_proxy="http://127.0.0.1:$proxy_port"
            export https_proxy="$http_proxy"
            echo -e "  ${GREEN}[√] Proxy configured: http://127.0.0.1:$proxy_port${NC}"
            break
        fi
    done
fi

# -----------------------------------------------------------------------------
# 4. Jupyter/WebSocket optimization (full configuration)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Configuring Jupyter WebSocket settings...${NC}"
mkdir -p ~/.jupyter

# Generate config if not exists
if [ ! -f ~/.jupyter/jupyter_server_config.py ]; then
    jupyter server --generate-config --confirm-exit 2>/dev/null || true
fi

# Write comprehensive Jupyter configuration
python3 -c '
import os
config_files = ["jupyter_server_config.py", "jupyter_notebook_config.py"]
jupyter_dir = os.path.expanduser("~/.jupyter")

config_block = """
# ========== Network & WebSocket Optimization for AutoDL ==========

# 1. WebSocket message size limit (prevents kernel disconnection on large outputs)
c.ServerApp.tornado_settings = {
    "websocket_max_message_size": 500 * 1024 * 1024,   # 500MB
    "websocket_ping_interval": 30000,                   # 30s ping interval (ms)
    "websocket_ping_timeout": 30000,                    # 30s timeout
}

# 2. Output rate limits (prevents kernel from being killed by excessive prints)
c.ServerApp.iopub_data_rate_limit = 10000000           # 10MB/s
c.ServerApp.rate_limit_window = 3.0                    # 3 seconds window
c.ServerApp.iopub_msg_rate_limit = 5000                # 5000 messages/second

# 3. Terminal session keep-alive
c.ServerApp.terminado_settings = {
    "inactive_timeout": 600,        # 10 minutes inactivity -> disconnect
    "ping_interval": 60,            # ping every 60 seconds
}

# 4. Kernel idle policy (keep long training alive)
c.MappingKernelManager.cull_idle_timeout = 86400       # 24 hours idle timeout
c.MappingKernelManager.cull_connected = False          # Don"t cull connected kernels
c.MappingKernelManager.cull_busy = False               # Don"t cull busy kernels

# 5. Cross-origin / proxy support (for trusted internal networks)
c.ServerApp.allow_origin = "*"
c.ServerApp.allow_remote_access = True
# c.ServerApp.disable_check_xsrf = True   # Only enable for internal networks!
"""

for fname in config_files:
    path = os.path.join(jupyter_dir, fname)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# Generated by Network Control Script\n")

    content = open(path).read()
    if "websocket_max_message_size" not in content:
        with open(path, "a") as f:
            f.write(config_block)
        print(f"  [√] Applied to {fname}")
    else:
        print(f"  [i] {fname} already optimized")
'

# -----------------------------------------------------------------------------
# 5. Terminal silencing and flow control
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Configuring terminal settings...${NC}"
BASHRC="$HOME/.bashrc"

# Disable Ctrl+S flow control and set colorful prompt
if ! grep -q "stty -ixon" "$BASHRC" 2>/dev/null; then
    cat <<'EOF' >> "$BASHRC"

# ========== Terminal Optimization for AutoDL ==========
# Disable Ctrl+S flow control (prevents accidental terminal freeze)
stty -ixon

# Colorful prompt: green user@host, blue path
export PS1="\[\e[32m\]\u@autodl\[\e[m\]:\[\e[34m\]\w\[\e[m\]\$ "

# Quick switch agent
alias proxy_on='export http_proxy=http://127.0.0.1:7890; export https_proxy=http://127.0.0.1:7890; echo "Proxy ON (127.0.0.1:7890)"'
alias proxy_off='unset http_proxy https_proxy; echo "Proxy OFF"'

# The optimized aria2 alias forces the use of a proxy and increases stability.
alias aria2p='aria2c --all-proxy="http://127.0.0.1:7890" --check-certificate=false -x 16 -s 16'

ulimit -n 65535
EOF
    echo -e "  ${GREEN}[√] Terminal settings added to ~/.bashrc${NC}"
else
    echo -e "  [i] Terminal settings already configured"
fi

# Apply settings to current session (if running interactively)
if [ -t 0 ]; then
    stty -ixon 2>/dev/null || true
    export PS1="\[\e[32m\]\u@autodl\[\e[m\]:\[\e[34m\]\w\[\e[m\]\$ "
fi

# -----------------------------------------------------------------------------
# 6. Kernel idle cleanup (already done in Jupyter config, just a reminder)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Kernel idle cleanup configuration...${NC}"
echo -e "  ${GREEN}[√] Idle timeout set to 24 hours (cull_idle_timeout=86400)${NC}"
echo -e "  ${GREEN}[√] Connected/Busy kernels will NOT be culled${NC}"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Network & Terminal optimization complete!${NC}"
echo -e "Note: 'iopub_data_rate_limit' has no effect on kernels that have already started.${NC}"
echo -e "Note: 'WebSocket size limit': The system must be shut down and then restarted (the instance must be restarted).${NC}"
echo -e "${YELLOW}Note: Please run 'source ~/.bashrc' and restart the kernel to apply prompt changes${NC}"
echo -e "${YELLOW}Note: Restart the kernel may quit from uv venv.${NC}"
echo -e "${BLUE}========================================${NC}"