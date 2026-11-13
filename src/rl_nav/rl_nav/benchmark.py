"""Benchmark RL vs Nav2 in Gazebo on the same goal sequence.

Both planners are driven through nav2_msgs/NavigateToPose:
  nav2 -> 'navigate_to_pose'      rl -> 'rl_navigate_to_pose'

The robot tours the goals in order (no teleporting, so AMCL stays valid). Per goal we log:
success, time, path length (from TF), minimum LiDAR clearance and a collision flag
(min scan < collision_range).

  ros2 run rl_nav benchmark --method nav2 --out results/nav2.csv
  ros2 run rl_nav benchmark --method rl   --out results/rl.csv
  ros2 run rl_nav benchmark --compare results/nav2.csv results/rl.csv
"""
import argparse
import csv
import json
import math
import os
import threading
import time

import yaml


def fix_straight_dist(csv_path, rows):
    """Older runs measured straight_dist to the goal centre; recompute start->end from the trajectory log."""
    traj_path = os.path.splitext(csv_path)[0] + '.traj.json'
    if not os.path.exists(traj_path):
        return rows
    with open(traj_path) as f:
        legs = json.load(f)['legs']
    for r, leg in zip(rows, legs):
        pts = [p for p in leg['traj'] if p]
        if len(pts) > 1:
            r['straight_dist'] = str(math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
    return rows


def compare(paths):
    import numpy as np
    rows = []
    for p in paths:
        with open(p) as f:
            r = fix_straight_dist(p, list(csv.DictReader(f)))
        ok = [x for x in r if x['success'] == '1']
        rows.append((os.path.basename(p), len(r),
                     np.mean([x['success'] == '1' for x in r]),
                     np.mean([x['collision'] == '1' for x in r]),
                     np.mean([float(x['path_length']) for x in ok]) if ok else float('nan'),
                     np.mean([float(x['straight_dist']) / max(float(x['path_length']), 1e-6) for x in ok]) if ok else float('nan'),
                     np.mean([float(x['time_s']) for x in ok]) if ok else float('nan'),
                     np.mean([float(x['min_clearance']) for x in r])))
    hdr = f'{"run":<20}{"goals":>6}{"success":>9}{"collide":>9}{"path_m":>9}{"eff":>7}{"time_s":>8}{"min_clr":>9}'
    print(hdr)
    print('-' * len(hdr))
    for n, g, s, c, pl, e, t, mc in rows:
        print(f'{n:<20}{g:>6}{s:>9.2f}{c:>9.2f}{pl:>9.2f}{e:>7.2f}{t:>8.1f}{mc:>9.2f}')


def run(args):
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener

    with open(args.goals) as f:
        cfg = yaml.safe_load(f)
    goals = cfg['goals'] * args.repeats

    rclpy.init()
    node = rclpy.create_node('nav_benchmark', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    state = {'min_scan': float('inf')}
    lock = threading.Lock()

    def on_scan(msg):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            with lock:
                state['min_scan'] = min(state['min_scan'], min(valid))

    node.create_subscription(LaserScan, 'scan', on_scan, qos_profile_sensor_data)
    action = 'navigate_to_pose' if args.method == 'nav2' else 'rl_navigate_to_pose'
    client = ActionClient(node, NavigateToPose, action)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    if not client.wait_for_server(timeout_sec=30):
        raise SystemExit(f'action server {action} not available')

    def pose():
        try:
            t = tf_buffer.lookup_transform(cfg.get('frame_id', 'map'), args.robot_frame, Time(),
                                           timeout=Duration(seconds=0.5)).transform.translation
            return t.x, t.y
        except Exception:
            return None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    traj_log = []   # per-goal trajectories -> <out>.traj.json (for plotting)
    fields = ['idx', 'goal_x', 'goal_y', 'success', 'collision', 'time_s', 'path_length',
              'straight_dist', 'min_clearance', 'status']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, g in enumerate(goals):
            start = pose()
            while start is None:
                time.sleep(0.2)
                start = pose()
            req = NavigateToPose.Goal()
            ps = PoseStamped()
            ps.header.frame_id = cfg.get('frame_id', 'map')
            ps.header.stamp = node.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y = float(g['x']), float(g['y'])
            yaw = float(g.get('yaw', 0.0))
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            req.pose = ps
            with lock:
                state['min_scan'] = float('inf')

            t0 = time.monotonic()                      # wall clock, for the timeout only
            sim_t0 = node.get_clock().now()            # sim time, for the reported leg time
            gh_future = client.send_goal_async(req)
            while not gh_future.done():
                time.sleep(0.05)
            gh = gh_future.result()
            path, last = 0.0, start
            traj = [start]
            status = GoalStatus.STATUS_UNKNOWN
            if gh.accepted:
                res_future = gh.get_result_async()
                while not res_future.done():
                    p = pose()
                    if p is not None:
                        path += math.hypot(p[0] - last[0], p[1] - last[1])
                        last = p
                        traj.append(p)
                    if time.monotonic() - t0 > args.timeout:
                        gh.cancel_goal_async()
                        break
                    time.sleep(0.1)
                if res_future.done():
                    status = res_future.result().status
                else:
                    time.sleep(1.0)
            elapsed = (node.get_clock().now() - sim_t0).nanoseconds * 1e-9
            end = pose() or last
            with lock:
                min_scan = state['min_scan']
            success = status == GoalStatus.STATUS_SUCCEEDED and \
                math.hypot(end[0] - g['x'], end[1] - g['y']) < args.success_radius
            row = dict(idx=i, goal_x=g['x'], goal_y=g['y'], success=int(success),
                       collision=int(min_scan < args.collision_range), time_s=round(elapsed, 2),
                       path_length=round(path, 3),
                       # straight line to where the robot actually stopped (it stops within the goal
                       # tolerance, so measuring to the goal centre would let efficiency exceed 1)
                       straight_dist=round(math.hypot(end[0] - start[0], end[1] - start[1]), 3),
                       min_clearance=round(min_scan, 3), status=status)
            w.writerow(row)
            f.flush()
            traj_log.append({'goal': [g['x'], g['y']], 'success': bool(success), 'traj': traj[::2]})
            with open(os.path.splitext(args.out)[0] + '.traj.json', 'w') as tf_:
                json.dump({'method': args.method, 'legs': traj_log}, tf_)
            print(row, flush=True)
    rclpy.try_shutdown()
    compare([args.out])


def main():
    from ament_index_python.packages import get_package_share_directory
    default_goals = os.path.join(get_package_share_directory('rl_nav'), 'config', 'benchmark_goals.yaml')
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['nav2', 'rl'])
    ap.add_argument('--goals', default=default_goals)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--out', default='results/benchmark.csv')
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--robot-frame', default='base_footprint')
    ap.add_argument('--success-radius', type=float, default=0.35)
    ap.add_argument('--collision-range', type=float, default=0.17,
                    help='LiDAR range treated as contact (waffle hull is ~0.1-0.2 m from the sensor)')
    ap.add_argument('--compare', nargs='+', metavar='CSV')
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.method:
        run(args)
    else:
        ap.error('give --method or --compare')


if __name__ == '__main__':
    main()
