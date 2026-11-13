"""Runs a trained PPO policy on a real/simulated TurtleBot3.

Exposes a nav2_msgs/NavigateToPose action server named `rl_navigate_to_pose`, so
the benchmark (and any Nav2 client) can drive RL and Nav2 through the same API.

Inputs : /scan (LaserScan), TF <global_frame> -> <robot_frame> (AMCL or odom)
Output : /cmd_vel (Twist)

Modes
  pure RL  (use_global_planner:=false): the policy sees the final goal directly.
  hybrid   (use_global_planner:=true) : Nav2's planner server (compute_path_to_pose) gives a
           global path every `replan_period_s`; the policy is fed a "carrot" `lookahead_m`
           ahead on that path, i.e. it replaces only Nav2's local controller (DWB).
"""
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from rl_nav.lidar_nav_env import RobotCfg, goal_features, lidar_to_obs


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RlPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy')
        self.declare_parameter('model_path', '')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('control_rate', 5.0)
        self.declare_parameter('timeout_s', 120.0)
        self.declare_parameter('scan_timeout_s', 1.0)
        self.declare_parameter('v_scale', 1.0)          # <1.0 to slow down on a real robot
        # Control in the odom frame (like Nav2's local controller): the robot pose comes from
        # odom -> base (smooth, never jumps) and only the goal is moved by AMCL's map -> odom.
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('use_odom_control', True)
        self.declare_parameter('use_global_planner', False)
        self.declare_parameter('lookahead_m', 0.8)
        self.declare_parameter('replan_period_s', 2.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        path = gp('model_path')
        if not path:
            raise RuntimeError('model_path parameter is required')
        from stable_baselines3 import PPO  # heavy import, after params
        self.policy = PPO.load(path, device='cpu')
        # robot limits the policy was trained with (saved by train_ppo next to the model)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), 'robot_cfg.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.rc = RobotCfg(**json.load(f))
        else:
            self.rc = RobotCfg()
            self.get_logger().warn(f'{cfg_path} not found, using default RobotCfg')
        self.odom_frame = gp('odom_frame')
        self.use_odom = bool(gp('use_odom_control'))
        self.global_frame, self.robot_frame = gp('global_frame'), gp('robot_frame')
        self.period = 1.0 / float(gp('control_rate'))
        self.timeout = float(gp('timeout_s'))
        self.scan_timeout = float(gp('scan_timeout_s'))
        self.v_scale = float(gp('v_scale'))
        self.use_planner = bool(gp('use_global_planner'))
        self.lookahead = float(gp('lookahead_m'))
        self.replan_period = float(gp('replan_period_s'))
        self._path = None            # Nx2 array, global frame
        self._path_lock = threading.Lock()

        self.cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._scan = None
        self._scan_lock = threading.Lock()
        self.create_subscription(LaserScan, 'scan', self._on_scan, qos_profile_sensor_data, callback_group=self.cb)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self._active = threading.Lock()
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose', callback_group=self.cb)
        self.server = ActionServer(
            self, NavigateToPose, 'rl_navigate_to_pose', execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.cb)
        self.get_logger().info(f'RL policy loaded from {path}; action server: rl_navigate_to_pose; '
                               f'w_max={self.rc.w_max}, control frame={self.odom_frame if self.use_odom else self.global_frame}, '
                               f'mode={"hybrid (Nav2 global planner + RL local)" if self.use_planner else "pure RL"}')

    def _on_scan(self, msg):
        with self._scan_lock:
            self._scan = msg

    def _on_goal(self, _):
        return GoalResponse.REJECT if self._active.locked() else GoalResponse.ACCEPT

    def _scan_to_360(self, msg: LaserScan) -> np.ndarray:
        """Resample an arbitrary LaserScan onto 360 beams at 1 deg starting at 0 rad (robot +x)."""
        r = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        idx = np.round(np.mod(ang, 2 * math.pi) / (2 * math.pi) * self.rc.lidar_beams).astype(int) % self.rc.lidar_beams
        out = np.full(self.rc.lidar_beams, np.inf, dtype=np.float32)
        r = np.where((r < msg.range_min) | (r > msg.range_max), np.inf, r)
        np.minimum.at(out, idx, r)
        return out

    def _lookup(self, target, source):
        tf = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.5))
        t = tf.transform.translation
        return t.x, t.y, _yaw(tf.transform.rotation)

    # ---------------------------------------------------------------- global planner (hybrid mode)
    def _request_path(self, gx, gy):
        if not self.planner.server_is_ready():
            return
        req = ComputePathToPose.Goal()
        req.goal.header.frame_id = self.global_frame
        req.goal.pose.position.x, req.goal.pose.position.y = gx, gy
        req.goal.pose.orientation.w = 1.0
        req.use_start = False                       # plan from the robot's current pose

        def on_result(fut):
            try:
                poses = fut.result().result.path.poses
            except Exception:
                return
            if poses:
                with self._path_lock:
                    self._path = np.array([[p.pose.position.x, p.pose.position.y] for p in poses])

        def on_goal(fut):
            gh = fut.result()
            if gh is not None and gh.accepted:
                gh.get_result_async().add_done_callback(on_result)

        self.planner.send_goal_async(req).add_done_callback(on_goal)

    def _carrot(self, gx, gy):
        """Point `lookahead_m` along the latest global path past the robot (global frame)."""
        with self._path_lock:
            path = self._path
        if path is None or len(path) < 2:
            return gx, gy
        try:
            rx, ry, _ = self._lookup(self.global_frame, self.robot_frame)
        except Exception:
            return gx, gy
        i = int(np.argmin(np.hypot(path[:, 0] - rx, path[:, 1] - ry)))
        acc = 0.0
        for j in range(i, len(path) - 1):
            acc += float(np.hypot(*(path[j + 1] - path[j])))
            if acc >= self.lookahead:
                return float(path[j + 1, 0]), float(path[j + 1, 1])
        return gx, gy                                # path end within lookahead -> the goal itself

    def _to_control(self, px, py):
        """Global-frame point -> control frame."""
        if not self.use_odom:
            return px, py
        ox, oy, oyaw = self._lookup(self.odom_frame, self.global_frame)
        c, s = math.cos(oyaw), math.sin(oyaw)
        return ox + c * px - s * py, oy + s * px + c * py

    def _pose_and_goal(self, gx, gy):
        """Robot pose and goal position, both in the control frame."""
        if not self.use_odom:
            return self._lookup(self.global_frame, self.robot_frame), (gx, gy)
        ox, oy, oyaw = self._lookup(self.odom_frame, self.global_frame)   # map expressed in odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        goal = (ox + c * gx - s * gy, oy + s * gx + c * gy)
        return self._lookup(self.odom_frame, self.robot_frame), goal

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _execute(self, goal_handle):
        with self._active:
            gx = goal_handle.request.pose.pose.position.x
            gy = goal_handle.request.pose.pose.position.y
            if goal_handle.request.pose.header.frame_id not in ('', self.global_frame):
                self.get_logger().warn(f'goal frame {goal_handle.request.pose.header.frame_id} != '
                                       f'{self.global_frame}; treating as {self.global_frame}')
            t0 = time.monotonic()
            rate = self.create_rate(1.0 / self.period)   # follows /clock when use_sim_time is set
            try:
                return self._control_loop(goal_handle, gx, gy, t0, rate)
            finally:
                self.destroy_rate(rate)
                self._stop()

    def _control_loop(self, goal_handle, gx, gy, t0, rate):
        prev_v = prev_w = 0.0
        last_plan = -1e9
        with self._path_lock:
            self._path = None
        fb = NavigateToPose.Feedback()
        result = NavigateToPose.Result()
        while rclpy.ok():
            tick = time.monotonic()
            if goal_handle.is_cancel_requested:
                self._stop()
                goal_handle.canceled()
                return result
            if tick - t0 > self.timeout:
                self._stop()
                self.get_logger().warn('RL goal timed out')
                goal_handle.abort()
                return result
            if self.use_planner and tick - last_plan > self.replan_period:
                self._request_path(gx, gy)
                last_plan = tick
            try:
                (x, y, yaw), (cgx, cgy) = self._pose_and_goal(gx, gy)
                tx, ty = (cgx, cgy)
                if self.use_planner:
                    tx, ty = self._to_control(*self._carrot(gx, gy))
            except Exception as e:
                self._stop()
                self.get_logger().warn(f'TF unavailable: {e}', throttle_duration_sec=2.0)
                rate.sleep()
                continue
            with self._scan_lock:
                scan = self._scan
            if scan is None or (self.get_clock().now() - Time.from_msg(scan.header.stamp)).nanoseconds * 1e-9 > self.scan_timeout:
                self._stop()
                self.get_logger().warn('no fresh scan', throttle_duration_sec=2.0)
                rate.sleep()
                continue

            dist = math.hypot(cgx - x, cgy - y)
            if dist < self.rc.goal_tolerance:
                self._stop()
                goal_handle.succeed()
                return result

            lid = lidar_to_obs(self._scan_to_360(scan), self.rc)
            gd, gs, gc = goal_features(tx - x, ty - y, yaw)    # carrot in hybrid mode, else the goal
            obs = np.concatenate([lid, [gd, gs, gc, prev_v / self.rc.v_max, prev_w / self.rc.w_max]]).astype(np.float32)
            act, _ = self.policy.predict(obs, deterministic=True)
            a = np.clip(act, -1, 1)
            v = self.rc.v_min + (a[0] + 1) * 0.5 * (self.rc.v_max - self.rc.v_min)
            w = a[1] * self.rc.w_max
            prev_v, prev_w = float(v), float(w)
            cmd = Twist()
            cmd.linear.x = float(v) * self.v_scale
            cmd.angular.z = float(w)
            self.cmd_pub.publish(cmd)

            fb.distance_remaining = float(dist)
            fb.current_pose.header.frame_id = self.odom_frame if self.use_odom else self.global_frame
            fb.current_pose.pose.position.x, fb.current_pose.pose.position.y = x, y
            goal_handle.publish_feedback(fb)
            rate.sleep()
        self._stop()
        goal_handle.abort()
        return result


def main():
    rclpy.init()
    node = RlPolicyNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
