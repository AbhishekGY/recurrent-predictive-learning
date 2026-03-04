"""
Image Dataset Collection for RPL Training

Collects passive episodes (zero force) and renders each state to a 64x64
grayscale image. Saves both images and ground-truth states (for linear
probing evaluation).

Usage:
    python -m pendulum.collect_image_data --num_episodes 2000 --output_path data/passive_swings_images.pkl
"""

import argparse
import os
import pickle
import tempfile

import numpy as np

from pendulum.environment import InvertedPendulum
from pendulum.render import render_pendulum


def save_atomic(obj, path: str) -> None:
    """Write *obj* to *path* atomically via temp-file + rename."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Collect passive image episodes for RPL training"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2000,
        help="Number of episodes to collect (default: 2000)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Maximum timesteps per episode (default: 500)",
    )
    parser.add_argument(
        "--theta_range", type=float, default=0.8,
        help="Max initial angle magnitude in radians (default: 0.8)",
    )
    parser.add_argument(
        "--omega_range", type=float, default=0.8,
        help="Max initial angular velocity magnitude in rad/s (default: 0.8)",
    )
    parser.add_argument(
        "--x_range", type=float, default=0.5,
        help="Max initial cart position in meters (default: 0.5)",
    )
    parser.add_argument(
        "--vx_range", type=float, default=0.3,
        help="Max initial cart velocity in m/s (default: 0.3)",
    )
    parser.add_argument(
        "--output_path", type=str, default="data/passive_swings_images.pkl",
        help="Output file path (default: data/passive_swings_images.pkl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = InvertedPendulum()

    episodes: list[dict] = []
    boundary_terminations = 0
    max_step_terminations = 0

    print(f"Collecting {args.num_episodes} passive image episodes ...")
    print(f"  theta range : [-{args.theta_range}, {args.theta_range}] rad")
    print(f"  omega range : [-{args.omega_range}, {args.omega_range}] rad/s")
    print(f"  x range     : [-{args.x_range}, {args.x_range}] m")
    print(f"  v_x range   : [-{args.vx_range}, {args.vx_range}] m/s")
    print(f"  max steps   : {args.max_steps}")
    print()

    for ep in range(args.num_episodes):
        state = env.reset(
            angle_range=args.theta_range,
            omega_range=args.omega_range,
            x_range=args.x_range,
            vx_range=args.vx_range,
        )
        states = [state.copy()]
        images = [render_pendulum(state)]

        done = False
        step = 0
        while not done and step < args.max_steps:
            state, done = env.step(0.0)
            states.append(state.copy())
            images.append(render_pendulum(state))
            step += 1

        episodes.append({
            "images": np.array(images, dtype=np.float32),  # (T+1, 1, 64, 64)
            "states": np.array(states, dtype=np.float32),   # (T+1, 4)
        })

        if done:
            boundary_terminations += 1
        else:
            max_step_terminations += 1

        if (ep + 1) % 100 == 0:
            print(f"  {ep + 1}/{args.num_episodes} episodes collected")

    # Save
    save_atomic(episodes, args.output_path)

    # Summary statistics
    lengths = [len(ep["states"]) - 1 for ep in episodes]
    all_states = np.concatenate([ep["states"] for ep in episodes], axis=0)

    print(f"\n{'=' * 50}")
    print(f"Collection complete — saved to {args.output_path}")
    print(f"{'=' * 50}")
    print(f"  Episodes:            {len(episodes)}")
    print(f"  Episode lengths:     mean={np.mean(lengths):.1f}  "
          f"min={np.min(lengths)}  max={np.max(lengths)}")
    print(f"  Termination:")
    print(f"    Cart boundary:     {boundary_terminations}")
    print(f"    Max steps reached: {max_step_terminations}")
    print(f"  State coverage:")
    labels = ["x (m)", "v_x (m/s)", "theta (rad)", "omega (rad/s)"]
    for i, label in enumerate(labels):
        col = all_states[:, i]
        print(f"    {label:15s}  min={col.min():+7.3f}  max={col.max():+7.3f}  "
              f"mean={col.mean():+7.3f}  std={col.std():.3f}")


if __name__ == "__main__":
    main()
