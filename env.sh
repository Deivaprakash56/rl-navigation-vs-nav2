# source ~/robot_learning_ws/env.sh
# ROS 2 Humble + this workspace + the project venv's Python packages (torch, SB3, mujoco, anthropic).
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
# venv was created with --system-site-packages, so prepend its site-packages for ROS nodes
export PYTHONPATH="$WS/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export PATH="$WS/.venv/bin:$PATH"
export TURTLEBOT3_MODEL=waffle
cd "$WS"
# TurtleBot3 Gazebo models (extracted from the ros-humble-turtlebot3-gazebo .deb, no sudo needed)
export GAZEBO_MODEL_PATH="$WS/third_party/tb3_gazebo/opt/ros/humble/share/turtlebot3_gazebo/models:/usr/share/gazebo-11/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
export GAZEBO_MODEL_DATABASE_URI=""   # don't block on the (dead) online model database
