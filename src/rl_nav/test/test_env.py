import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_nav.lidar_nav_env import LidarNavEnv, RobotCfg, World, lidar_to_obs


def test_gym_api():
    check_env(LidarNavEnv(), skip_render_check=True)


def test_raycast_wall_and_circle():
    w = World(10.0, np.array([[5.0, 3.0, 0.5]]), np.zeros((0, 4)))
    r = w.raycast(5.0, 5.0, np.array([0.0, math.pi / 2, -math.pi / 2]), 20.0)
    assert abs(r[0] - 5.0) < 1e-6          # east wall
    assert abs(r[1] - 5.0) < 1e-6          # north wall
    assert abs(r[2] - 1.5) < 1e-6          # circle surface at y=3.5


def test_raycast_box():
    w = World(10.0, np.zeros((0, 3)), np.array([[6.0, 4.0, 7.0, 6.0]]))
    r = w.raycast(5.0, 5.0, np.array([0.0, math.pi]), 20.0)
    assert abs(r[0] - 1.0) < 1e-6 and abs(r[1] - 5.0) < 1e-6


def test_lidar_sector_front():
    rc = RobotCfg()
    scan = np.full(360, np.inf)
    scan[358] = 1.0          # just right of straight ahead -> must land in sector 0
    obs = lidar_to_obs(scan, rc)
    assert abs(obs[0] - 1.0 / rc.lidar_range_max) < 1e-6 and np.all(obs[1:] == 1.0)


def test_episode_runs():
    env = LidarNavEnv()
    obs, _ = env.reset(seed=1)
    for _ in range(500):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            assert 'is_success' in info
            break
