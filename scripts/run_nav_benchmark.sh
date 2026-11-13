#!/bin/bash
# Project 2: benchmark the PPO policy against Nav2 in Gazebo on the same goal tour.
# Each method gets a fresh simulation (same launch file, robot spawned at the same pose).
# Usage: scripts/run_nav_benchmark.sh runs/ppo_v2/best_model.zip [repeats] [methods] [rl_name]
#   methods: "nav2 rl" (default) or just "rl" to reuse an existing results/nav2.csv
#   rl_name: output name for the RL run, e.g. rl_v2 -> results/rl_v2.csv
#   HYBRID=true scripts/run_nav_benchmark.sh ...   -> RL as local controller on Nav2's global path
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=$(realpath "${1:-$WS/runs/ppo_v1/best_model.zip}")
REPEATS=${2:-2}
METHODS=${3:-"nav2 rl"}
RL_NAME=${4:-rl}
source "$WS/env.sh" >/dev/null
# run on our own ROS domain so another simulation on this machine cannot cross-talk (/clock, /tf, /scan)
export ROS_DOMAIN_ID=${SIM_DOMAIN_ID:-77}
export GAZEBO_MASTER_URI=http://localhost:${SIM_GAZEBO_PORT:-11377}   # separate gzserver master
mkdir -p "$WS/results" "$WS/logs"
trap 'bash "$WS/scripts/kill_sim.sh" >/dev/null' EXIT

for METHOD in $METHODS; do
  NAME=$METHOD; [ "$METHOD" = "rl" ] && NAME=$RL_NAME
  bash "$WS/scripts/kill_sim.sh"
  LOG="$WS/logs/p2_sim_$METHOD.log"
  ros2 launch rl_nav rl_nav_sim.launch.py model_path:="$MODEL" use_global_planner:=${HYBRID:-false} use_rviz:=False > "$LOG" 2>&1 &
  echo "[$METHOD] waiting for Nav2 + RL policy node ..."
  timeout 180 bash -c "until grep -q 'AMCL initialised' '$LOG' && grep -q 'RL policy loaded' '$LOG' && \
      grep -q 'lifecycle_manager_navigation.*Managed nodes are active' '$LOG'; do
    grep -q 'gzserver-1\]: process has died' '$LOG' && { echo 'gzserver died'; exit 1; }; sleep 2; done"
  sleep 5
  ros2 run rl_nav benchmark --method "$METHOD" --repeats "$REPEATS" --out "$WS/results/$NAME.csv"
done
bash "$WS/scripts/kill_sim.sh"
ros2 run rl_nav benchmark --compare "$WS/results/nav2.csv" "$WS/results/$RL_NAME.csv"
python3 -m rl_nav.plot_results benchmark "$WS/results/nav2.csv" "$WS/results/$RL_NAME.csv" --out "$WS/src/rl_nav/docs"
