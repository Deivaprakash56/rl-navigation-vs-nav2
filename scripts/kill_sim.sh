#!/bin/bash
# Stop the Gazebo / Nav2 / project nodes started from THIS workspace - and nothing else.
# A launch counts as ours if its environment references this workspace's install/ directory;
# we then signal that launch process and all of its descendants (gzserver, nav2_container, ...).
# Other ROS sessions on the machine (another workspace, another terminal) are left untouched.
WS="$(cd "$(dirname "$0")/.." && pwd)"

our_launches() {
  for pid in $(pgrep -f "ros2 launch (lang_nav|rl_nav)"); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q "$WS/install" && echo "$pid"
  done
}
descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }

ROOTS=$(our_launches)
[ -z "$ROOTS" ] && { echo "sim stopped (nothing of ours running)"; exit 0; }
ALL="$ROOTS"
for r in $ROOTS; do ALL="$ALL $(descendants "$r")"; done

kill -INT $ROOTS 2>/dev/null; sleep 3
kill -TERM $ALL 2>/dev/null
for i in $(seq 1 16); do
  alive=""; for p in $ALL; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
  [ -z "$alive" ] && { echo "sim stopped"; exit 0; }
  [ "$i" -eq 8 ] && kill -KILL $alive 2>/dev/null
  sleep 0.5
done
echo "still running:$alive"; exit 1
