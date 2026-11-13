"""Fast 2D LiDAR navigation environment (Gymnasium) for PPO training.

Why not train in Gazebo directly: PPO needs ~1-5M steps; Gazebo runs ~1x real time,
this env runs thousands of steps/s per core. The policy is then evaluated (and
fine-tuned if needed) in Gazebo through the ROS node, using the *same* observation
function (`lidar_to_obs`) so there is no train/deploy mismatch.

Robot: TurtleBot3 Waffle-like differential drive (radius 0.22 m, v<=0.26 m/s, w<=1.82 rad/s).
LiDAR: 360 beams over 2*pi starting at the robot's +x axis, CCW (same as TB3 /scan),
       range [0.12, 3.5] m, min-pooled into N sectors.

Observation (float32, all roughly in [-1, 1]):
  [ sector ranges / max_range  (n_sectors)
    goal distance / 5 (clipped to 1), sin(goal bearing), cos(goal bearing),
    previous v (normalised), previous w (normalised) ]
Action: [v, w] in [-1, 1]^2 -> v in [v_min, v_max], w in [-w_max, w_max]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RobotCfg:
    radius: float = 0.22
    v_max: float = 0.26
    v_min: float = -0.05          # tiny reverse allowed to escape tight spots
    w_max: float = 1.82
    lidar_range_min: float = 0.12
    lidar_range_max: float = 3.5
    lidar_beams: int = 360
    n_sectors: int = 24
    dt: float = 0.2               # control period (TB3 scan arrives at 5 Hz in Gazebo)
    goal_tolerance: float = 0.25


@dataclass
class EnvCfg:
    robot: RobotCfg = field(default_factory=RobotCfg)
    max_steps: int = 400
    room_size: tuple = (5.0, 10.0)
    n_obstacles: tuple = (4, 14)
    goal_dist: tuple = (1.0, 6.0)
    # domain randomisation for sim-to-real
    lidar_noise_std: float = 0.02
    lidar_dropout_p: float = 0.01   # beams returning inf
    action_noise_std: float = 0.03
    latency_steps_max: int = 1      # actions applied 0..N steps late
    # reward weights
    r_progress: float = 10.0
    r_goal: float = 20.0
    r_collision: float = -20.0
    r_step: float = -0.02
    r_turn: float = -0.02
    r_near_obstacle: float = -0.1   # per-step penalty, ramps up as clearance drops below safety_margin
    safety_margin: float = 0.22     # metres of free space wanted around the robot body


def cfg_presets(name: str) -> 'EnvCfg':
    """v1: TB3 hardware limits (w 1.82 rad/s), weak obstacle shaping - the first iteration.
    v2: Nav2 DWB limits (w 1.0 rad/s) and a 0.25 m safety margin with 3x the penalty."""
    if name == 'v1':
        return EnvCfg()
    if name == 'v2':
        return EnvCfg(robot=RobotCfg(w_max=1.0), r_near_obstacle=-0.3, safety_margin=0.25)
    raise ValueError(name)


def lidar_to_obs(ranges: np.ndarray, cfg: RobotCfg) -> np.ndarray:
    """Raw 360-beam scan (inf/nan allowed) -> normalised sector minima. Shared by sim and ROS."""
    r = np.asarray(ranges, dtype=np.float32).copy()
    r[~np.isfinite(r)] = cfg.lidar_range_max
    r[r < cfg.lidar_range_min] = cfg.lidar_range_max  # below min range = invalid return on TB3
    r = np.clip(r, 0.0, cfg.lidar_range_max)
    n = len(r)
    # sector k is centred on angle 2*pi*k/n_sectors; roll so sector 0 is centred on the front beam
    per = n // cfg.n_sectors
    rolled = np.roll(r, per // 2)
    sectors = rolled[: per * cfg.n_sectors].reshape(cfg.n_sectors, per).min(axis=1)
    return sectors / cfg.lidar_range_max


def goal_features(dx: float, dy: float, yaw: float):
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - yaw
    return min(dist / 5.0, 1.0), math.sin(bearing), math.cos(bearing)


class World:
    """Square room + random circles and axis-aligned boxes."""

    def __init__(self, size, circles, boxes):
        self.size = size                    # room spans [0, size]^2
        self.circles = circles              # (N,3) cx, cy, r
        self.boxes = boxes                  # (M,4) xmin, ymin, xmax, ymax

    def clearance(self, x, y) -> float:
        """Distance from point to nearest obstacle surface or wall."""
        d = min(x, y, self.size - x, self.size - y)
        if len(self.circles):
            d = min(d, float(np.min(np.hypot(self.circles[:, 0] - x, self.circles[:, 1] - y) - self.circles[:, 2])))
        if len(self.boxes):
            b = self.boxes
            dx = np.maximum(np.maximum(b[:, 0] - x, 0), x - b[:, 2])
            dy = np.maximum(np.maximum(b[:, 1] - y, 0), y - b[:, 3])
            outside = np.hypot(dx, dy)
            inside = (dx == 0) & (dy == 0)
            outside[inside] = -1.0
            d = min(d, float(outside.min()))
        return d

    def raycast(self, x, y, angles, max_range) -> np.ndarray:
        """Vectorised ray distances for all angles."""
        cx, cy = np.cos(angles), np.sin(angles)
        t = np.full(angles.shape, max_range, dtype=np.float64)
        # walls (box [0,size]^2 seen from inside)
        with np.errstate(divide='ignore', invalid='ignore'):
            for bound, comp, pos in ((0.0, cx, x), (self.size, cx, x), (0.0, cy, y), (self.size, cy, y)):
                tw = (bound - pos) / comp
                tw[(tw <= 0) | ~np.isfinite(tw)] = np.inf
                t = np.minimum(t, tw)
        # circles
        if len(self.circles):
            ox = self.circles[:, 0][None] - x
            oy = self.circles[:, 1][None] - y
            r = self.circles[:, 2][None]
            b = ox * cx[:, None] + oy * cy[:, None]
            c = ox ** 2 + oy ** 2 - r ** 2
            disc = b ** 2 - c
            tc = b - np.sqrt(np.maximum(disc, 0))
            tc[(disc < 0) | (tc <= 0)] = np.inf
            t = np.minimum(t, tc.min(axis=1))
        # boxes (slab method)
        if len(self.boxes):
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_x = 1.0 / cx[:, None]
                inv_y = 1.0 / cy[:, None]
                bx0 = (self.boxes[:, 0][None] - x) * inv_x
                bx1 = (self.boxes[:, 2][None] - x) * inv_x
                by0 = (self.boxes[:, 1][None] - y) * inv_y
                by1 = (self.boxes[:, 3][None] - y) * inv_y
            tmin = np.maximum(np.minimum(bx0, bx1), np.minimum(by0, by1))
            tmax = np.minimum(np.maximum(bx0, bx1), np.maximum(by0, by1))
            hit = (tmax >= tmin) & (tmax > 0) & np.isfinite(tmin)
            tb = np.where(hit, np.where(tmin > 0, tmin, 0.0), np.inf)
            tb = np.nan_to_num(tb, nan=np.inf)
            t = np.minimum(t, tb.min(axis=1))
        return t


class LidarNavEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 5}

    def __init__(self, cfg: EnvCfg | None = None, render_mode=None):
        self.cfg = cfg or EnvCfg()
        rc = self.cfg.robot
        self.render_mode = render_mode
        n_obs = rc.n_sectors + 5
        self.observation_space = spaces.Box(-1.0, 1.0, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._beam_angles = np.arange(rc.lidar_beams) * (2 * math.pi / rc.lidar_beams)
        self.world = None

    # ------------------------------------------------------------------ helpers
    def scale_action(self, a):
        rc = self.cfg.robot
        a = np.clip(a, -1, 1)
        v = rc.v_min + (a[0] + 1) * 0.5 * (rc.v_max - rc.v_min)
        w = a[1] * rc.w_max
        return float(v), float(w)

    def _sample_free(self, margin):
        for _ in range(500):
            p = self.np_random.uniform(margin, self.world.size - margin, 2)
            if self.world.clearance(*p) > margin:
                return p
        return None

    def _make_world(self):
        c = self.cfg
        rng = self.np_random
        size = rng.uniform(*c.room_size)
        n = rng.integers(c.n_obstacles[0], c.n_obstacles[1] + 1)
        n_circ = rng.integers(0, n + 1)
        circles = np.column_stack([rng.uniform(0, size, n_circ), rng.uniform(0, size, n_circ),
                                   rng.uniform(0.1, 0.5, n_circ)]) if n_circ else np.zeros((0, 3))
        n_box = n - n_circ
        if n_box:
            cxy = rng.uniform(0, size, (n_box, 2))
            half = rng.uniform(0.1, 0.8, (n_box, 2))
            boxes = np.column_stack([cxy - half, cxy + half])
        else:
            boxes = np.zeros((0, 4))
        self.world = World(size, circles, boxes)

    def _scan(self):
        c = self.cfg
        rc = c.robot
        r = self.world.raycast(self.x, self.y, self._beam_angles + self.yaw, rc.lidar_range_max)
        r = r + self.np_random.normal(0, c.lidar_noise_std, r.shape)
        r[r > rc.lidar_range_max] = np.inf  # TB3 reports inf when nothing is hit
        drop = self.np_random.random(r.shape) < c.lidar_dropout_p
        r[drop] = np.inf
        return r

    def _obs(self):
        rc = self.cfg.robot
        self.last_scan = self._scan()
        lid = lidar_to_obs(self.last_scan, rc)
        gd, gs, gc = goal_features(self.goal[0] - self.x, self.goal[1] - self.y, self.yaw)
        pv = self.prev_v / rc.v_max
        pw = self.prev_w / rc.w_max
        return np.concatenate([lid, [gd, gs, gc, pv, pw]]).astype(np.float32)

    # ------------------------------------------------------------------ gym api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        rc = c.robot
        for _ in range(50):
            self._make_world()
            s = self._sample_free(rc.radius + 0.15)
            g = None
            if s is not None:
                for _ in range(200):
                    g = self._sample_free(rc.radius + 0.1)
                    if g is not None and c.goal_dist[0] <= np.linalg.norm(g - s) <= c.goal_dist[1]:
                        break
                    g = None
            if s is not None and g is not None:
                break
        self.x, self.y = float(s[0]), float(s[1])
        self.yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.goal = g
        self.prev_v = self.prev_w = 0.0
        self.steps = 0
        self.latency = int(self.np_random.integers(0, c.latency_steps_max + 1))
        self.action_queue = [(0.0, 0.0)] * self.latency
        self.path_len = 0.0
        self.prev_dist = float(np.linalg.norm(self.goal - [self.x, self.y]))
        self.start_dist = self.prev_dist
        self.trajectory = [(self.x, self.y)]
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        rc = c.robot
        action = np.asarray(action, dtype=np.float32) + self.np_random.normal(0, c.action_noise_std, 2)
        cmd = self.scale_action(action)
        self.action_queue.append(cmd)
        v, w = self.action_queue.pop(0)

        # integrate unicycle with sub-steps for collision accuracy
        n_sub = 4
        h = rc.dt / n_sub
        collided = False
        for _ in range(n_sub):
            nx = self.x + v * math.cos(self.yaw) * h
            ny = self.y + v * math.sin(self.yaw) * h
            self.yaw = (self.yaw + w * h + math.pi) % (2 * math.pi) - math.pi
            self.path_len += math.hypot(nx - self.x, ny - self.y)
            self.x, self.y = nx, ny
            if self.world.clearance(self.x, self.y) < rc.radius:
                collided = True
                break
        self.trajectory.append((self.x, self.y))
        self.steps += 1
        self.prev_v, self.prev_w = cmd

        dist = float(np.linalg.norm(self.goal - [self.x, self.y]))
        reward = c.r_progress * (self.prev_dist - dist) + c.r_step + c.r_turn * abs(w) / rc.w_max
        self.prev_dist = dist
        clearance = self.world.clearance(self.x, self.y) - rc.radius
        if clearance < c.safety_margin:
            reward += c.r_near_obstacle * (1 - clearance / c.safety_margin)

        terminated = truncated = False
        success = False
        if collided:
            reward += c.r_collision
            terminated = True
        elif dist < rc.goal_tolerance:
            reward += c.r_goal
            terminated = success = True
        elif self.steps >= c.max_steps:
            truncated = True

        info = {}
        if terminated or truncated:
            x0, y0 = self.trajectory[0]
            info = {'is_success': success, 'collision': collided, 'path_length': self.path_len,
                    'start_dist': self.start_dist, 'steps': self.steps,
                    'displacement': math.hypot(self.x - x0, self.y - y0)}   # straight line start -> stop
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
        draw_episode(ax, self)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img


def draw_episode(ax, env: LidarNavEnv):
    import matplotlib.patches as mp
    w = env.world
    ax.add_patch(mp.Rectangle((0, 0), w.size, w.size, fill=False, lw=2))
    for cx, cy, r in w.circles:
        ax.add_patch(mp.Circle((cx, cy), r, color='0.4'))
    for x0, y0, x1, y1 in w.boxes:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, color='0.4'))
    tr = np.array(env.trajectory)
    ax.plot(tr[:, 0], tr[:, 1], 'b-', lw=1.5)
    ax.add_patch(mp.Circle((env.x, env.y), env.cfg.robot.radius, fill=False, color='b'))
    ax.plot(*env.goal, 'g*', ms=14)
    ax.set_xlim(-0.2, w.size + 0.2)
    ax.set_ylim(-0.2, w.size + 0.2)
    ax.set_aspect('equal')


gym.register(id='LidarNav-v0', entry_point='rl_nav.lidar_nav_env:LidarNavEnv')
gym.register(id='LidarNav-v2', entry_point='rl_nav.lidar_nav_env:LidarNavEnv', kwargs={'cfg': cfg_presets('v2')})
