#!/usr/bin/env bash
# Run ON the always-on machine, from the repo root:  ./deploy/install-service.sh
# Sets up the venv, a launchd service that starts at login and restarts on
# crash, and a Tailscale HTTPS address for it.
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
LABEL="com.morningdrill.web"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${PORT:-8000}"

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements-web.txt

# Shared secret so only you can POST code, even on a shared tailnet.
[ -f .drill_token ] || { openssl rand -hex 24 > .drill_token; chmod 600 .drill_token; }
TOKEN="$(cat .drill_token)"

mkdir -p "$HOME/Library/LaunchAgents" logs
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>uvicorn</string>
    <string>web.app:app</string><string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>$PORT</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>DRILL_EXEC_MODE</key><string>subprocess</string>
    <key>DRILL_AUTH_TOKEN</key><string>$TOKEN</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/web.out.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/web.err.log</string>
</dict></plist>
PLISTEOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
tailscale serve --bg "$PORT"

echo
echo "Running. Open the https://...ts.net address above with ?token=$TOKEN once per device."
echo "Logs: $ROOT/logs/web.err.log   Stop: launchctl bootout gui/$(id -u)/$LABEL"
