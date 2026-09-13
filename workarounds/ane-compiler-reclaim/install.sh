#!/bin/sh
# Install the root LaunchDaemon. Run as: sudo sh install.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
label=local.ane-compiler-reclaim
plist=/Library/LaunchDaemons/$label.plist
script=/usr/local/libexec/ane-compiler-reclaim.sh
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
plutil -lint "$here/$label.plist"
sh -n "$here/ane-compiler-reclaim.sh"
# Root runs this script, so it must be root-owned and not writable by the user.
install -d -o root -g wheel -m 755 /usr/local/libexec
install -o root -g wheel -m 755 "$here/ane-compiler-reclaim.sh" "$script"
install -o root -g wheel -m 644 "$here/$label.plist" "$plist"
launchctl bootout "system/$label" 2>/dev/null || true
launchctl bootstrap system "$plist"
launchctl print "system/$label" | grep -E 'state =|run interval|last exit code'
echo "current decision:"
ANE_RECLAIM_DRY_RUN=1 sh "$script"
