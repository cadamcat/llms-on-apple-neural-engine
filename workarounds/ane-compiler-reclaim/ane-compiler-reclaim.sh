#!/bin/sh
# Restart ANECompilerService when it holds deleted MetalPerformanceShadersGraph compile inputs.
# The service keeps each ANE-region .mlir open after the client process has exited and the
# file has been unlinked, so the blocks stay allocated until the service itself exits.
# Runs as root from /Library/LaunchDaemons/local.ane-compiler-reclaim.plist.
set -u
PATH=/usr/bin:/bin:/usr/sbin:/sbin
log=${ANE_RECLAIM_LOG:-/Library/Logs/ane-compiler-reclaim.log}
min_bytes=${ANE_RECLAIM_MIN_BYTES:-2147483648}
busy=${ANE_RECLAIM_BUSY_PROCESS:-powermetrics}
scratch_glob=${ANE_RECLAIM_SCRATCH_GLOB:-/private/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph/mpsgraph-*}
dry_run=${ANE_RECLAIM_DRY_RUN:-0}

skip() {
  [ "$dry_run" = 1 ] && echo "decision=skip reason=$1"
  exit 0
}
alive() { ps -p "$1" >/dev/null 2>&1; }

# Skip while the configured measurement process is detected to reduce interference.
pgrep -x "$busy" >/dev/null 2>&1 && skip measurement_running

# A live process with MPSGraph scratch may be about to hand a new input to the service.
for dir in $scratch_glob; do
  [ -d "$dir" ] || continue
  pid=${dir##*/mpsgraph-}
  pid=${pid%%-*}
  case "$pid" in ''|*[!0-9]*) continue ;; esac
  alive "$pid" && skip "scratch_owner_alive pid=$pid"
done

if [ -n "${ANE_RECLAIM_LSOF_INPUT:-}" ]; then
  listing=$(cat "$ANE_RECLAIM_LSOF_INPUT")
else
  listing=$(lsof -nP -c ANECompiler -F pskn 2>/dev/null)
fi

# One line per MPSGraph file the service holds: service pid, input owner pid, link count, bytes.
records=$(printf '%s\n' "$listing" | awk '
function flush() {
  if (name ~ /\/com\.apple\.MetalPerformanceShadersGraph\/mpsgraph-[0-9]+-/) {
    owner = name; sub(/.*\/mpsgraph-/, "", owner); sub(/-.*/, "", owner)
    print pid, owner, (links == "" ? "?" : links), (size == "" ? 0 : size)
  }
  name = ""; size = ""; links = ""
}
/^p/ { flush(); pid = substr($0, 2); next }
/^f/ { flush(); next }
/^s/ { size = substr($0, 2); next }
/^k/ { links = substr($0, 2); next }
/^n/ { name = substr($0, 2); next }
END { flush() }')
[ -n "$records" ] || skip no_held_scratch

services=""
held_files=0
held_bytes=0
while read -r service owner links size; do
  alive "$owner" && skip "input_owner_alive pid=$owner"
  [ "$links" = 0 ] || continue
  held_files=$((held_files + 1))
  held_bytes=$((held_bytes + size))
  case " $services " in *" $service "*) ;; *) services="$services $service" ;; esac
done <<EOF
$records
EOF
services=${services# }
[ "$held_bytes" -ge "$min_bytes" ] || skip "below_min held_bytes=$held_bytes"

if [ "$dry_run" = 1 ]; then
  echo "decision=reclaim services=$services held_files=$held_files held_bytes=$held_bytes"
  exit 0
fi

free_kib() { df -k /System/Volumes/Data | awk 'NR == 2 { print $4 }'; }
before=$(free_kib)
signals=""
for service in $services; do
  case "$(ps -p "$service" -o comm= 2>/dev/null)" in *ANECompilerService) ;; *) continue ;; esac
  kill -TERM "$service" 2>/dev/null
  i=0
  while alive "$service" && [ "$i" -lt 10 ]; do sleep 1; i=$((i + 1)); done
  signal=TERM
  if alive "$service"; then kill -KILL "$service" 2>/dev/null; signal=KILL; fi
  signals="$signals $service:$signal"
done
sleep 5
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) reclaimed signals=${signals# } held_files=$held_files held_bytes=$held_bytes free_kib_before=$before free_kib_after=$(free_kib)" >> "$log"
