#!/usr/bin/env bash
# run_logged.sh LOG STATUS COMMAND [ARGS...]
#
# Run COMMAND with its output in LOG, then write its exit code to STATUS
# (atomically, so a reader never sees a partial file). Used by the Clients
# page's Upgrade button, which starts upgrade.sh in a transient systemd unit
# and reads the result back from these files.
set -u

if [[ $# -lt 3 ]]; then
  echo "usage: $0 LOG STATUS COMMAND [ARGS...]" >&2
  exit 2
fi

log=$1
status=$2
shift 2

"$@" >"$log" 2>&1 </dev/null
code=$?
printf '%s\n' "$code" >"$status.tmp" && mv -f "$status.tmp" "$status"
exit "$code"
