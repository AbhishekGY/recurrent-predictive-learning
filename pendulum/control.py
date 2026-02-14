"""
Predictive Control Script for Inverted Pendulum

Uses the trained RPL model's learned forward model to select control actions.
At each timestep, evaluates candidate forces by predicting their outcomes
in latent space and choosing the one closest to the goal (upright, centered).

Usage:
    python -m pendulum.control --checkpoint checkpoints/rpl_model_final.pt
"""

import argparse

import numpy as np
import torch

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel


class PredictiveController:
    """
    One-step lookahead controller using RPL's learned forward model.

    Evaluates candidate forces by predicting their outcomes in latent space,
    then selects the action that brings the predicted next state closest
    to the goal embedding (upright, centered pendulum).
    """

    def __init__(
        self,
        model: RPLModel,
        candidate_forces: list[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.candidate_forces = candidate_forces or [-10.0, -5.0, 0.0, 5.0, 10.0]

        # Compute goal embedding: upright, centered, stationary
        goal_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
        with torch.no_grad():
            self.z_goal = self.model.encoder(goal_state)  # (1, 32)

        # Persistent LSTM hidden state
        self.hidden = None

    def reset(self):
        """Reset the controller's hidden state for a new episode."""
        self.hidden = None

    def select_action(self, state: np.ndarray) -> float:
        """
        Select the best control action for the current state.

        Evaluates each candidate force by predicting the resulting next
        state embedding, then picks the one closest to the goal.

        Args:
            state: Current state as numpy array [x, v_x, theta, omega]

        Returns:
            Best force to apply (float)
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        costs = []
        with torch.no_grad():
            for force in self.candidate_forces:
                force_tensor = torch.tensor([[force]], dtype=torch.float32, device=self.device)

                # Predict next embedding for this hypothetical action
                # Uses current hidden state but does NOT persist the result
                z_next_pred = self.model.predict_for_action(
                    state_tensor, force_tensor, self.hidden
                )

                # Cost = distance to goal in latent space
                cost = torch.norm(z_next_pred - self.z_goal) ** 2
                costs.append(cost.item())

        # Select action with lowest cost
        best_idx = np.argmin(costs)
        return self.candidate_forces[best_idx]

    def update(self, state: np.ndarray, action: float):
        """
        Update the persistent hidden state with the actual outcome.

        Call this AFTER executing the action in the real environment.

        Args:
            state: The state that resulted from executing the action
            action: The force that was actually applied
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([[action]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, _, self.hidden = self.model.forward_step(
                state_tensor, action_tensor, self.hidden
            )


def run_episode(
    env: InvertedPendulum,
    controller: PredictiveController,
    max_steps: int = 500,
    angle_range: float = 0.05,
    omega_range: float = 0.05,
) -> dict:
    """
    Run a single control episode.

    Args:
        env: The pendulum environment
        controller: The predictive controller
        max_steps: Maximum timesteps per episode
        angle_range: Initial angle perturbation
        omega_range: Initial angular velocity perturbation

    Returns:
        Dictionary with episode data
    """
    state = env.reset(angle_range=angle_range, omega_range=omega_range)
    controller.reset()

    states = [state.copy()]
    forces = []

    done = False
    step = 0

    while not done and step < max_steps:
        # Select action using the learned model
        force = controller.select_action(state)
        forces.append(force)

        # Execute in the real environment
        state, done = env.step(force)
        states.append(state.copy())

        # Update persistent hidden state with actual outcome
        controller.update(state, force)

        step += 1

    return {
        'states': np.array(states),
        'forces': np.array(forces),
        'length': step,
        'survived': not done,
    }


def run_random_baseline(
    env: InvertedPendulum,
    num_episodes: int = 50,
    max_steps: int = 500,
    angle_range: float = 0.05,
    omega_range: float = 0.05,
) -> list[int]:
    """Run episodes with random control as a baseline."""
    lengths = []
    for _ in range(num_episodes):
        env.reset(angle_range=angle_range, omega_range=omega_range)
        done = False
        step = 0
        while not done and step < max_steps:
            force = np.random.uniform(-10, 10)
            _, done = env.step(force)
            step += 1
        lengths.append(step)
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Run predictive control on inverted pendulum")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rpl_model_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of control episodes to run (default: 50)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--angle_range",
        type=float,
        default=0.05,
        help="Initial angle perturbation (default: 0.05 rad)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== RPL Predictive Control ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Initial angle range: ±{args.angle_range} rad")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Create controller and environment
    controller = PredictiveController(model, device=device)
    env = InvertedPendulum()

    # Run random baseline
    print(f"\nRunning random baseline ({args.num_episodes} episodes)...")
    random_lengths = run_random_baseline(
        env, args.num_episodes, args.max_steps,
        args.angle_range, args.angle_range,
    )

    print(f"Random control:")
    print(f"  Mean: {np.mean(random_lengths):.1f} steps ({np.mean(random_lengths) * 0.02:.2f}s)")
    print(f"  Max:  {np.max(random_lengths)} steps ({np.max(random_lengths) * 0.02:.2f}s)")

    # Run RPL controller
    print(f"\nRunning RPL controller ({args.num_episodes} episodes)...")
    rpl_lengths = []
    for i in range(args.num_episodes):
        result = run_episode(
            env, controller, args.max_steps,
            args.angle_range, args.angle_range,
        )
        rpl_lengths.append(result['length'])

        if (i + 1) % 10 == 0:
            recent = rpl_lengths[-10:]
            print(f"  Episodes {i-8:3d}-{i+1:3d}: "
                  f"mean={np.mean(recent):.0f} steps, "
                  f"max={np.max(recent)} steps")

    # Summary
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"{'Metric':<25} {'Random':<15} {'RPL Controller'}")
    print("-" * 55)
    print(f"{'Mean steps':<25} {np.mean(random_lengths):>8.1f}       {np.mean(rpl_lengths):>8.1f}")
    print(f"{'Max steps':<25} {np.max(random_lengths):>8d}       {np.max(rpl_lengths):>8d}")
    print(f"{'Min steps':<25} {np.min(random_lengths):>8d}       {np.min(rpl_lengths):>8d}")
    print(f"{'Mean time (s)':<25} {np.mean(random_lengths)*0.02:>8.2f}       {np.mean(rpl_lengths)*0.02:>8.2f}")

    improvement = np.mean(rpl_lengths) / max(np.mean(random_lengths), 1)
    print(f"\nImprovement: {improvement:.1f}x longer balance time")

    if improvement > 2:
        print("✓ RPL controller significantly outperforms random control!")
    elif improvement > 1.2:
        print("~ RPL controller shows moderate improvement")
    else:
        print("✗ RPL controller needs more training to show clear improvement")


if __name__ == "__main__":
    main()
