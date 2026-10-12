#!/usr/bin/env bash
# Run ON the always-on machine:  ./deploy/update.sh
# Pulls the latest code, refreshes dependencies, restarts the service, and
# checks that it answers.
set -e
cd "$(dirname "$0")/.."
LABEL="com.morningdrill.web"
PORT="${PORT:-8000}"

git pull --ff-only
.venv/bin/pip install -q -r requirements-web.txt
if [ -f "/Library/LaunchDaemons/$LABEL.plist" ]; then
  sudo launchctl kickstart -k "system/$LABEL"      # installed with install-daemon.sh
else
  launchctl kickstart -k "gui/$(id -u)/$LABEL"     # installed with install-service.sh
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || true)"
  [ "$code" = "200" ] && { echo "Updated to $(git log --oneline -1). Serving (200)."; exit 0; }
done

echo "Restarted, but no 200 from http://127.0.0.1:$PORT/ (last: ${code:-none})." >&2
echo "See: tail -20 $PWD/logs/web.err.log" >&2
exit 1
