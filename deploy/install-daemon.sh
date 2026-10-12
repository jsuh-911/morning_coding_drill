#!/usr/bin/env bash
# Run ON the always-on machine, as the admin user, from the repo root:
#   ./deploy/install-daemon.sh
#
# Replaces the per-login agent with a system daemon: it starts at boot and keeps
# running when nobody is logged in. It still runs as YOU, not root. Asks for
# your password once (sudo), to write into /Library/LaunchDaemons.
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
ME="$(id -un)"
LABEL="com.morningdrill.web"
PLIST="/Library/LaunchDaemons/$LABEL.plist"
PORT="${PORT:-8000}"

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements-web.txt

# Same token as before, so the URLs you already use keep working.
[ -f .drill_token ] || { openssl rand -hex 24 > .drill_token; chmod 600 .drill_token; }
TOKEN="$(cat .drill_token)"

mkdir -p logs
touch logs/web.out.log logs/web.err.log

# Remove the per-login agent first, or two copies fight for the port.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"

TMP="$(mktemp)"
cat > "$TMP" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>UserName</key><string>$ME</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>uvicorn</string>
    <string>web.app:app</string><string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>$PORT</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>HOME</key><string>$HOME</string>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    <key>DRILL_EXEC_MODE</key><string>subprocess</string>
    <key>DRILL_AUTH_TOKEN</key><string>$TOKEN</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/web.out.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/web.err.log</string>
</dict></plist>
PLISTEOF

sudo install -m 644 -o root -g wheel "$TMP" "$PLIST"
rm -f "$TMP"

sudo launchctl bootout "system/$LABEL" 2>/dev/null || true
sudo launchctl bootstrap system "$PLIST"
tailscale serve --bg "$PORT"

echo
echo "Installed as a system daemon running as $ME. It starts at boot and survives logout."
echo "Logs: $ROOT/logs/web.err.log"
echo "Stop: sudo launchctl bootout system/$LABEL"
