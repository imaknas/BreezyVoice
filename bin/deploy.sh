#!/usr/bin/env bash
# Update a production checkout to origin/main and restart the service that runs from it.
# The service never runs from a development clone, so editing one never changes what is being served.
# Commit and push first; this deploys what is on the remote.
#
#   bin/deploy.sh            # pull, sync, test, restart
#   bin/deploy.sh --force    # deploy even while a client is in the middle of a job
#
# Where things live on a given machine is not part of the repository. Put it in `deploy.env` next to
# this checkout's README (ignored by git), or in the environment:
#
#   BREEZYVOICE_RUNTIME=/path/to/production/checkout     (required)
#   BREEZYVOICE_SERVICE_LABEL=launchd.label.of.service   (required)
#   BREEZYVOICE_PORT=8765                                (default 8765)
#   BREEZYVOICE_BUSY_PATTERN='pgrep -f pattern'          (optional: processes that must not be interrupted)
#
# Steps: refuse while a client is busy (a restart drops an in-flight request) → clone on first run →
# fast-forward to origin/main → uv sync --locked → full test suite incl. the smoke test (synthesizes one
# sentence) → restart the service → wait for /health. Any failure stops before the restart, so the running
# service keeps the previous code.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HERE/deploy.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/deploy.env"
  set +a
fi
: "${BREEZYVOICE_RUNTIME:?set BREEZYVOICE_RUNTIME (see the header of this script)}"
: "${BREEZYVOICE_SERVICE_LABEL:?set BREEZYVOICE_SERVICE_LABEL (see the header of this script)}"
RUNTIME="$BREEZYVOICE_RUNTIME"
LABEL="$BREEZYVOICE_SERVICE_LABEL"
PORT="${BREEZYVOICE_PORT:-8765}"
REPO="$(git -C "$HERE" remote get-url origin)"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$FORCE" != 1 && -n "${BREEZYVOICE_BUSY_PATTERN:-}" ]] && pgrep -f "$BREEZYVOICE_BUSY_PATTERN" >/dev/null; then
  echo "❌ a client of the service is running and may be mid-request; try later or pass --force" >&2
  exit 75
fi

if [[ ! -d "$RUNTIME/.git" ]]; then
  echo "first deploy: cloning into $RUNTIME"
  git clone -q "$REPO" "$RUNTIME"
fi
cd "$RUNTIME"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "❌ $RUNTIME has local changes; production is only changed by this script" >&2
  exit 1
fi
git fetch -q origin
git checkout -q main
git merge --ff-only -q origin/main
uv sync --locked -q
HF_HUB_OFFLINE=1 uv run --locked pytest -q -p no:cacheprovider

if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
  for _ in $(seq 1 60); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
      echo "✅ $(git log -1 --oneline) — $LABEL healthy on :$PORT"
      exit 0
    fi
    sleep 2
  done
  echo "❌ $LABEL did not become healthy within 120 s; see $RUNTIME/serve.log" >&2
  exit 1
fi
echo "✅ $(git log -1 --oneline) — $LABEL is not loaded yet"
