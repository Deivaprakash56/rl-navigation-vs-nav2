"""README figures for rl_nav.

  python -m rl_nav.plot_results training  runs/ppo_v1             --out docs/
  python -m rl_nav.plot_results obs                                --out docs/
  python -m rl_nav.plot_results benchmark results/nav2.csv results/rl.csv --out docs/
"""
import argparse
import csv
import glob
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-2, surface and ink tokens
BLUE, ORANGE = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
AQUA = '#1baf7a'
# colour follows the entity (file stem), never its position in the plot
# categorical slots in fixed order (reference palette); colour follows the entity, never its rank
METHOD_COLOR = {'nav2': ORANGE, 'rl': BLUE, 'rl_v1': AQUA, 'rl_v1_odom': '#eda100', 'rl_v2': '#e87ba4',
                'rl_v2_hybrid': BLUE, 'rl_v1_hybrid': '#6250d6'}
METHOD_NAME = {'nav2': 'Nav2', 'rl': 'PPO', 'rl_v1': 'PPO v1', 'rl_v1_odom': 'PPO v1 + odom', 'rl_v2': 'PPO v2',
               'rl_v2_hybrid': 'Nav2 planner + PPO v2', 'rl_v1_hybrid': 'Nav2 planner + PPO v1'}


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'lines.linewidth': 2,
        'legend.frameon': False})


# ----------------------------------------------------------------------------- training
def _tb_scalars(run_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    xs, ys = [], []
    for d in sorted(glob.glob(os.path.join(run_dir, 'PPO_*'))):
        ea = EventAccumulator(d, size_guidance={'scalars': 0})
        ea.Reload()
        if tag in ea.Tags().get('scalars', []):
            for e in ea.Scalars(tag):
                xs.append(e.step)
                ys.append(e.value)
    return np.array(xs), np.array(ys)


def training(run_dir, out):
    ev = np.load(os.path.join(run_dir, 'evaluations.npz'))
    t = ev['timesteps'] / 1e6
    succ = ev['successes'].mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    x, y = _tb_scalars(run_dir, 'rollout/success_rate_win')
    if len(x):
        ax.plot(x / 1e6, y, color=BLUE, lw=1.2, alpha=0.45, label='training episodes (rolling 200)')
    ax.plot(t, succ, color=BLUE, marker='o', ms=4, label='evaluation (40 eps, deterministic)')
    best = int(np.argmax(ev['results'].mean(axis=1)))   # SB3 keeps the best *mean reward* checkpoint
    ax.plot(t[best], succ[best], marker='o', ms=11, mfc='none', mec=INK, mew=1.5)
    ax.annotate('saved model\n(best mean reward)', (t[best], succ[best]), textcoords='offset points',
                xytext=(-95, -40), fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('environment steps (millions)')
    ax.set_title('Success rate')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc='lower right', fontsize=8.5)

    ax = axes[1]
    x, y = _tb_scalars(run_dir, 'rollout/collision_rate_win')
    if len(x):
        ax.plot(x / 1e6, y, color=BLUE, lw=1.6)
    ax.set_ylim(0, max(0.5, float(y.max()) * 1.1) if len(y) else 1)
    ax.set_xlabel('environment steps (millions)')
    ax.set_title('Collision rate (training episodes, rolling 200)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    p = os.path.join(out, 'training_curves.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


# ----------------------------------------------------------------------------- observation
def obs_figure(out, seed=7):
    import matplotlib.patches as mp
    from rl_nav.lidar_nav_env import LidarNavEnv, draw_episode
    env = LidarNavEnv()
    env.cfg.lidar_noise_std = 0.0
    env.cfg.lidar_dropout_p = 0.0
    obs, _ = env.reset(seed=seed)
    rc = env.cfg.robot
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    draw_episode(ax, env)
    scan = env.last_scan
    ang = env._beam_angles + env.yaw
    r = np.where(np.isfinite(scan), scan, rc.lidar_range_max)
    for a, d in zip(ang[::3], r[::3]):
        ax.plot([env.x, env.x + d * math.cos(a)], [env.y, env.y + d * math.sin(a)], color=BLUE, lw=0.4, alpha=0.5)
    ax.annotate('', xy=tuple(env.goal), xytext=(env.x, env.y),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.8, ls='--'))
    ax.set_title('World: robot, 360 LiDAR beams, goal')
    ax.grid(False)

    ax = fig.add_subplot(1, 3, 2, projection='polar')
    n = rc.n_sectors
    centers = np.arange(n) * 2 * math.pi / n
    vals = obs[:n] * rc.lidar_range_max
    ax.bar(centers, vals, width=2 * math.pi / n * 0.9, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=1)
    ax.plot(np.radians(np.arange(360)), r, color=INK2, lw=0.8, alpha=0.7)
    ax.set_theta_zero_location('E')
    ax.set_ylim(0, rc.lidar_range_max)
    ax.set_title('Robot frame: raw scan (line) → 24 sector minima (bars)', pad=14)
    ax.tick_params(labelsize=8)

    ax = fig.add_subplot(1, 3, 3)
    labels = [f's{i}' for i in range(n)] + ['goal dist', 'sin θg', 'cos θg', 'prev v', 'prev w']
    colors = [BLUE] * n + [ORANGE] * 3 + ['#1baf7a'] * 2
    ax.barh(range(len(obs)), obs, color=colors, height=0.75)
    ax.set_yticks(range(len(obs)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_title('Policy input: 29 numbers in [-1, 1]')
    ax.legend(handles=[mp.Patch(color=BLUE, label='LiDAR sectors / 3.5 m'), mp.Patch(color=ORANGE, label='goal'),
                       mp.Patch(color='#1baf7a', label='previous action')], fontsize=8, loc='lower right')
    fig.tight_layout()
    p = os.path.join(out, 'observation.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


# ----------------------------------------------------------------------------- rollout gif
def rollout_gif(model_path, out, seeds=(3, 11, 21, 42), cfg='v2'):
    import imageio
    from stable_baselines3 import PPO
    from rl_nav.lidar_nav_env import LidarNavEnv, cfg_presets, draw_episode
    policy = PPO.load(model_path, device='cpu')
    envs, obs, done, info = [], [], [], [{} for _ in seeds]
    for s in seeds:
        e = LidarNavEnv(cfg_presets(cfg))
        o, _ = e.reset(seed=s)
        envs.append(e)
        obs.append(o)
        done.append(False)
    frames = []
    for t in range(400):
        if t % 2 == 0 or all(done):
            fig, axes = plt.subplots(1, len(envs), figsize=(3.2 * len(envs), 3.4), dpi=70)
            for ax, e, d, inf in zip(axes, envs, done, info):
                draw_episode(ax, e)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                tag = ('reached' if inf.get('is_success') else 'collision' if inf.get('collision') else 'timeout') \
                    if d else f'step {e.steps}'
                ax.set_title(tag, fontsize=10, color=INK2 if not d else (BLUE if inf.get('is_success') else ORANGE))
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        if all(done):
            break
        for i, e in enumerate(envs):
            if not done[i]:
                a, _ = policy.predict(obs[i], deterministic=True)
                obs[i], _, term, trunc, inf = e.step(a)
                if term or trunc:
                    done[i], info[i] = True, inf
    frames += [frames[-1]] * 12
    p = os.path.join(out, 'rollouts_2d.gif')
    imageio.mimsave(p, frames, duration=0.1, loop=0)
    print('wrote', p, '| outcomes:', [i.get('is_success') for i in info])


# ----------------------------------------------------------------------------- benchmark
def _load(csv_path):
    from rl_nav.benchmark import fix_straight_dist
    with open(csv_path) as f:
        rows = fix_straight_dist(csv_path, list(csv.DictReader(f)))
    traj_path = os.path.splitext(csv_path)[0] + '.traj.json'
    traj = json.load(open(traj_path)) if os.path.exists(traj_path) else None
    method = os.path.splitext(os.path.basename(csv_path))[0]   # file stem: nav2, rl_v1, rl_v2
    return method, rows, traj


def benchmark(csvs, out, map_yaml='/opt/ros/humble/share/nav2_bringup/maps/turtlebot3_world.yaml', tag=''):
    import yaml
    data = [_load(c) for c in csvs]

    # map overlay
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img = plt.imread(os.path.join(os.path.dirname(map_yaml), meta['image']))
    h, w = img.shape[:2]
    ox, oy = meta['origin'][:2]
    ext = (ox, ox + w * meta['resolution'], oy, oy + h * meta['resolution'])
    ncols = min(3, len(data))
    nrows = int(math.ceil(len(data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(data):]:
        ax.axis('off')
    for ax, (method, rows, traj) in zip(axes, data):
        ax.imshow(img, cmap='gray', extent=ext, origin='upper')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect('equal')
        ax.grid(False)
        c = METHOD_COLOR.get(method, BLUE)
        if traj:
            for i, leg in enumerate(traj['legs']):
                tr = np.array([p for p in leg['traj'] if p])
                if len(tr) > 1:
                    ax.plot(tr[:, 0], tr[:, 1], color=c, lw=1.6, alpha=0.85)
                gx, gy = leg['goal']
                ax.plot(gx, gy, marker='o' if leg['success'] else 'X', ms=9, mfc=c if leg['success'] else 'white',
                        mec=INK, mew=1.2)
                if i < 10:
                    ax.text(gx + 0.1, gy + 0.1, str(i), fontsize=8, color=INK2)
        ax.set_xticks([])
        ax.set_yticks([])
        n_ok = sum(r['success'] == '1' for r in rows)
        ax.set_title(f'{METHOD_NAME.get(method, method)}: {n_ok}/{len(rows)} goals reached')
    fig.tight_layout()
    p = os.path.join(out, f'benchmark_paths{tag}.png')
    fig.savefig(p, dpi=120)
    print('wrote', p)

    # metric small multiples
    def m(rows, key, only_success=True):
        rs = [r for r in rows if r['success'] == '1'] if only_success else rows
        return [float(r[key]) for r in rs]

    metrics = [
        ('Success rate', lambda rows: np.mean([r['success'] == '1' for r in rows]), True),
        ('Near-contact rate\n(LiDAR < 0.17 m)', lambda rows: np.mean([r['collision'] == '1' for r in rows]), True),
        ('Path length [m]\n(successful legs)', lambda rows: np.mean(m(rows, 'path_length')), False),
        ('Path efficiency\n(successful legs)',
         lambda rows: np.mean([float(r['straight_dist']) / max(float(r['path_length']), 1e-6)
                               for r in rows if r['success'] == '1']), False),
        ('Time [s]\n(successful legs)', lambda rows: np.mean(m(rows, 'time_s')), False),
        ('Min LiDAR clearance [m]\n(all legs)', lambda rows: np.mean(m(rows, 'min_clearance', False)), False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(2.3, 0.75 * len(data) + 1.2) * len(metrics), 3.6))
    for ax, (title, fn, pct) in zip(axes, metrics):
        names, vals, cols = [], [], []
        for method, rows, _ in data:
            names.append(METHOD_NAME.get(method, method))
            vals.append(fn(rows))
            cols.append(METHOD_COLOR.get(method, BLUE))
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.0%}' if pct else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=9, color=INK)
        ax.set_xticks(range(len(vals)))
        rot = 35 if len(vals) > 2 else 0
        ax.set_xticklabels(names, fontsize=8 if rot else 8.5, rotation=rot, ha='right' if rot else 'center')
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis='x', visible=False)
        if pct:
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    fig.tight_layout()
    p = os.path.join(out, f'benchmark_metrics{tag}.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['training', 'obs', 'benchmark', 'gif'])
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--out', default='docs')
    ap.add_argument('--cfg', default='v2', choices=['v1', 'v2'])
    ap.add_argument('--tag', default='', help='suffix for benchmark figure names')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()
    if args.what == 'training':
        training(args.inputs[0], args.out)
    elif args.what == 'gif':
        rollout_gif(args.inputs[0], args.out, cfg=args.cfg)
    elif args.what == 'obs':
        obs_figure(args.out)
    else:
        benchmark(args.inputs, args.out, tag=args.tag)


if __name__ == '__main__':
    main()
