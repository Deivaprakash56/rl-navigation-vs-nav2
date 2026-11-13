"""Evaluate a trained policy in the 2D env and plot trajectories.

  python -m rl_nav.eval_2d runs/ppo_v2/best_model.zip --episodes 300 --cfg v2 --plot traj.png
  python -m rl_nav.eval_2d random --episodes 50     # random-policy sanity baseline
"""
import argparse

import numpy as np

from rl_nav.lidar_nav_env import LidarNavEnv, cfg_presets, draw_episode


def line_blocked(env, n=60):
    """True if the straight start->goal segment passes closer than the robot radius to an obstacle."""
    (sx, sy), (gx, gy) = env.trajectory[0], env.goal
    return any(env.world.clearance(sx + (gx - sx) * t, sy + (gy - sy) * t) < env.cfg.robot.radius
               for t in np.linspace(0, 1, n))


def run(policy, episodes, seed, plot=None, cfg='v2'):
    env = LidarNavEnv(cfg_presets(cfg))
    stats = {'success': [], 'collision': [], 'timeout': [], 'path_eff': [], 'steps': [], 'blocked': []}
    snapshots = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        blocked = line_blocked(env)
        done = False
        while not done:
            act = env.action_space.sample() if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        stats['success'].append(info['is_success'])
        stats['collision'].append(info['collision'])
        stats['timeout'].append(trunc and not term)
        stats['steps'].append(info['steps'])
        stats['blocked'].append(blocked)
        if info['is_success']:
            stats['path_eff'].append(info['displacement'] / max(info['path_length'], 1e-6))
        if plot and len(snapshots) < 12 and (blocked or len(snapshots) < 4):
            import copy
            tag = 'reached goal' if info['is_success'] else ('collision' if info['collision'] else 'timeout')
            snapshots.append((copy.deepcopy(env), tag + (' · line of sight blocked' if blocked else ' · clear line')))
    print(f'episodes          : {episodes}')
    print(f'success rate      : {np.mean(stats["success"]):.3f}')
    print(f'collision rate    : {np.mean(stats["collision"]):.3f}')
    print(f'timeout rate      : {np.mean(stats["timeout"]):.3f}')
    if stats['path_eff']:
        print(f'path efficiency   : {np.mean(stats["path_eff"]):.3f}  (straight line start->stop / travelled, successes only)')
    print(f'mean steps        : {np.mean(stats["steps"]):.1f}')
    b = np.array(stats['blocked'])
    sc = np.array(stats['success'])
    cc = np.array(stats['collision'])
    if b.any():
        print(f'blocked line-of-sight episodes: {b.sum()} -> success {sc[b].mean():.3f}, collision {cc[b].mean():.3f}')
    if (~b).any():
        print(f'clear line-of-sight episodes  : {(~b).sum()} -> success {sc[~b].mean():.3f}, collision {cc[~b].mean():.3f}')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for ax, (e, tag) in zip(axes.flat, snapshots):
            draw_episode(ax, e)
            ax.set_title(tag, fontsize=11, color='#2a78d6' if tag.startswith('reached') else '#d03b3b')
        fig.tight_layout()
        fig.savefig(plot)
        print(f'wrote {plot}')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='SB3 .zip, or "random"')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=10_000)
    ap.add_argument('--plot')
    ap.add_argument('--cfg', default='v2', choices=['v1', 'v2'], help='env preset the policy was trained with')
    args = ap.parse_args()
    policy = None
    if args.model != 'random':
        from stable_baselines3 import PPO
        policy = PPO.load(args.model, device='cpu')
    run(policy, args.episodes, args.seed, args.plot, args.cfg)


if __name__ == '__main__':
    main()
