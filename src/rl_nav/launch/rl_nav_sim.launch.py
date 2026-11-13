"""Gazebo TurtleBot3 world + Nav2 (AMCL localisation + Nav2 planner) + RL policy action server.

  ros2 launch rl_nav rl_nav_sim.launch.py model_path:=$HOME/robot_learning_ws/runs/ppo_v1/best_model.zip

Then in another terminal:
  ros2 run rl_nav benchmark --method rl   --out results/rl.csv
  ros2 run rl_nav benchmark --method nav2 --out results/nav2.csv   (restart sim between runs)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav2_dir = get_package_share_directory('nav2_bringup')
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'tb3_simulation_launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless'),
                          'use_rviz': LaunchConfiguration('use_rviz')}.items())
    init_pose = Node(package='rl_nav', executable='set_initial_pose', output='screen',
                     parameters=[{'use_sim_time': True, 'x': -2.0, 'y': -0.5, 'yaw': 0.0}])
    rl = Node(package='rl_nav', executable='rl_policy_node', name='rl_policy', output='screen',
              parameters=[{'model_path': LaunchConfiguration('model_path'), 'use_sim_time': True,
                           'use_global_planner': LaunchConfiguration('use_global_planner')}])
    return LaunchDescription([
        DeclareLaunchArgument('model_path'),
        DeclareLaunchArgument('use_global_planner', default_value='false',
                              description='hybrid: Nav2 global path + RL local controller'),
        DeclareLaunchArgument('headless', default_value='True'),
        DeclareLaunchArgument('use_rviz', default_value='True'),
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        sim,
        TimerAction(period=5.0, actions=[init_pose]),
        TimerAction(period=8.0, actions=[rl]),
    ])
