"""
Data Collection Script for RPL Training

Runs random exploration episodes with the inverted pendulum environment
and saves the collected state-action trajectories for training.

Usage:
    python -m pendulum.collect_data --num_episodes 2000 --output_path data/random_exploration.pkl
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from pendulum.environment import InvertedPendulum


def collect_episode(
    env: InvertedPendulum,
    max_steps: int,
    force_range: float = 10.0,
) -> dict:
    """
    Collect a single episode with random actions.

    Args:
        env: The pendulum environment
        max_steps: Maximum number of timesteps per episode
        force_range: Maximum force magnitude (samples from [-force_range, force_range])

    Returns:
        Dictionary with:
            - 'states': np.array of shape (T+1, 4) - all states including initial
            - 'forces': np.array of shape (T, 1) - forces applied at each step
    """
    states = []
    forces = []

    # Get initial state
    state = env.get_state()
    states.append(state.copy())

    done = False
    step = 0

    while not done and step < max_steps:
        # Sample random force
        force = np.random.uniform(-force_range, force_range)
        forces.append([force])

        # Step environment
        state, done = env.step(force)
        states.append(state.copy())

        step += 1

    return {
        'states': np.array(states, dtype=np.float32),  # (T+1, 4)
        'forces': np.array(forces, dtype=np.float32),  # (T, 1)
    }


def collect_dataset(
    num_episodes: int,
    max_steps: int,
    angle_range: float,
    omega_range: float,
    x_range: float = 0.0,
    vx_range: float = 0.0,
    seed: int = 42,
) -> list:
    """
    Collect multiple episodes of random exploration.

    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        angle_range: Initial angle perturbation range
        omega_range: Initial angular velocity range
        x_range: Initial cart position range
        vx_range: Initial cart velocity range
        seed: Random seed for reproducibility

    Returns:
        List of episode dictionaries
    """
    np.random.seed(seed)

    env = InvertedPendulum()
    episodes = []

    for i in range(num_episodes):
        # Reset with random initial conditions
        env.reset(
            angle_range=angle_range,
            omega_range=omega_range,
            x_range=x_range,
            vx_range=vx_range,
        )

        # Collect episode
        episode = collect_episode(env, max_steps)
        episodes.append(episode)

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1}/{num_episodes} episodes")

    return episodes


def print_stats(episodes: list) -> None:
    """Print summary statistics about the collected episodes."""
    lengths = [len(ep['forces']) for ep in episodes]

    print("\n=== Dataset Statistics ===")
    print(f"Number of episodes: {len(episodes)}")
    print(f"Episode lengths:")
    print(f"  Mean: {np.mean(lengths):.1f} steps")
    print(f"  Min:  {np.min(lengths)} steps")
    print(f"  Max:  {np.max(lengths)} steps")
    print(f"  Std:  {np.std(lengths):.1f} steps")

    total_transitions = sum(lengths)
    print(f"Total transitions: {total_transitions:,}")

    # State statistics from all episodes
    all_states = np.concatenate([ep['states'] for ep in episodes], axis=0)
    print(f"\nState statistics (x, v_x, theta, omega):")
    print(f"  Mean: {all_states.mean(axis=0)}")
    print(f"  Std:  {all_states.std(axis=0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect random exploration data for RPL training"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2000,
        help="Number of episodes to collect (default: 2000)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/random_exploration.pkl",
        help="Output path for the dataset (default: data/random_exploration.pkl)",
    )
    parser.add_argument(
        "--angle_range",
        type=float,
        default=0.2,
        help="Initial angle perturbation range in radians (default: 0.2)",
    )
    parser.add_argument(
        "--omega_range",
        type=float,
        default=0.2,
        help="Initial angular velocity range in rad/s (default: 0.2)",
    )
    parser.add_argument(
        "--x_range",
        type=float,
        default=0.0,
        help="Initial cart position range (default: 0.0)",
    )
    parser.add_argument(
        "--vx_range",
        type=float,
        default=0.0,
        help="Initial cart velocity range (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=== Random Exploration Data Collection ===")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Initial angle range: ±{args.angle_range} rad")
    print(f"Initial omega range: ±{args.omega_range} rad/s")
    print(f"Initial x range: ±{args.x_range} m")
    print(f"Initial vx range: ±{args.vx_range} m/s")
    print(f"Output path: {args.output_path}")
    print(f"Random seed: {args.seed}")
    print()

    # Collect data
    episodes = collect_dataset(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        angle_range=args.angle_range,
        omega_range=args.omega_range,
        x_range=args.x_range,
        vx_range=args.vx_range,
        seed=args.seed,
    )

    # Print statistics
    print_stats(episodes)

    # Create output directory if needed
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    with open(output_path, 'wb') as f:
        pickle.dump(episodes, f)

    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
