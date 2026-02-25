"""
Passive Data Collection for RPL Inverted Pendulum

Collects episodes of a pendulum swinging freely under gravity with zero applied
force.  Initial conditions are sampled broadly so the dataset covers a wide
range of dynamics (full swings, near-upright oscillations, etc.).

Episodes terminate only when the cart hits the track boundary or the maximum
number of steps is reached — angle is never a termination condition.

Usage:
    python -m pendulum.collect_passive_data --num_episodes 3000
"""

import argparse
import os
import pickle
import tempfile

import numpy as np

from pendulum.environment import InvertedPendulum


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
        description="Collect passive swing data for RPL representation learning"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3000,
        help="Number of episodes to collect (default: 3000)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Maximum timesteps per episode (default: 500)",
    )
    parser.add_argument(
        "--output_path", type=str, default="data/passive_swings.pkl",
        help="Output file path (default: data/passive_swings.pkl)",
    )
    parser.add_argument(
        "--theta_range", type=float, default=1.0,
        help="Max initial angle magnitude in radians (default: 1.0)",
    )
    parser.add_argument(
        "--omega_range", type=float, default=1.0,
        help="Max initial angular velocity magnitude in rad/s (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: None)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    env = InvertedPendulum()

    episodes: list[dict] = []
    boundary_terminations = 0
    max_step_terminations = 0

    print(f"Collecting {args.num_episodes} passive episodes ...")
    print(f"  theta range : [-{args.theta_range}, {args.theta_range}] rad")
    print(f"  omega range : [-{args.omega_range}, {args.omega_range}] rad/s")
    print(f"  x range     : [-0.5, 0.5] m")
    print(f"  v_x range   : [-0.3, 0.3] m/s")
    print(f"  max steps   : {args.max_steps}")
    print()

    for ep in range(args.num_episodes):
        state = env.reset(
            angle_range=args.theta_range,
            omega_range=args.omega_range,
            x_range=0.5,
            vx_range=0.3,
        )
        states = [state.copy()]

        done = False
        step = 0
        while not done and step < args.max_steps:
            state, done = env.step(0.0)
            states.append(state.copy())
            step += 1

        episodes.append({
            "states": np.array(states, dtype=np.float32),  # (T+1, 4)
        })

        if done:
            boundary_terminations += 1
        else:
            max_step_terminations += 1

        if (ep + 1) % 500 == 0:
            print(f"  {ep + 1}/{args.num_episodes} episodes collected")

    # Save
    save_atomic(episodes, args.output_path)

    # Summary statistics
    lengths = [len(ep["states"]) - 1 for ep in episodes]  # number of transitions
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
