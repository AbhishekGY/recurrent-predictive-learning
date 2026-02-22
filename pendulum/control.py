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
    Random shooting controller using RPL's learned forward model.

    Samples N random force sequences of length H, rolls each out through
    the learned forward model, accumulates cost across all H steps, and
    picks the first action of the lowest-cost sequence.
    """

    def __init__(
        self,
        model: RPLModel,
        candidate_forces: list[float] = None,
        horizon: int = 5,
        num_samples: int = 200,
        cart_penalty_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.candidate_forces = candidate_forces or [-10.0, -5.0, 0.0, 5.0, 10.0]
        self.horizon = horizon
        self.num_samples = num_samples
        self.cart_penalty_weight = cart_penalty_weight

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
        Select the best control action via batched random shooting.

        Samples num_samples random force sequences of length horizon,
        rolls all of them out simultaneously through the learned forward
        model, accumulates cost, and returns the first action of the
        lowest-cost sequence.

        Args:
            state: Current state as numpy array [x, v_x, theta, omega]

        Returns:
            Best force to apply (float)
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 4)

        with torch.no_grad():
            # Encode current state once
            z_current = self.model.encoder(state_tensor)  # (1, 32)

            # Sample all force sequences at once
            # shape: (num_samples, horizon)
            force_sequences = torch.FloatTensor(
                self.num_samples, self.horizon
            ).uniform_(-10, 10).to(self.device)

            # Expand z_current and hidden state to batch size num_samples
            z = z_current.expand(self.num_samples, -1)  # (num_samples, 32)

            # Expand hidden state to match batch size
            if self.hidden is None:
                h = None
            else:
                h = (
                    self.hidden[0].expand(-1, self.num_samples, -1).contiguous(),
                    self.hidden[1].expand(-1, self.num_samples, -1).contiguous(),
                )

            # Expand goal to match batch size
            z_goal = self.z_goal.expand(self.num_samples, -1)  # (num_samples, 32)

            total_costs = torch.zeros(self.num_samples, device=self.device)

            # Roll out all samples in parallel across the horizon
            for t in range(self.horizon):
                forces_t = force_sequences[:, t].unsqueeze(1)  # (num_samples, 1)

                h_out, h = self.model.integrator(z, forces_t, h)  # h_out: (num_samples, 64)
                z = self.model.predictor(h_out)  # (num_samples, 32)

                # Accumulate cost for all samples simultaneously
                cost = ((z - z_goal) ** 2).sum(dim=1)  # (num_samples,)
                total_costs += cost

            # Cart penalty: bias toward actions that push cart back toward center
            x_current = state[0]
            vx_current = state[1]

            # Penalize first force that has same sign as cart position
            # (i.e., pushes cart further from center)
            first_forces = force_sequences[:, 0]  # (num_samples,)
            cart_penalty = self.cart_penalty_weight * (
                x_current * first_forces +  # penalize forces in same direction as x
                0.1 * vx_current * first_forces  # penalize forces in same direction as vx
            )
            total_costs += cart_penalty

            # Pick the first force of the lowest-cost trajectory
            best_idx = total_costs.argmin()
            best_first_force = force_sequences[best_idx, 0].item()

        return best_first_force

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

    # Determine termination cause
    final_state = states[-1]
    if not done:
        termination = 'survived'
    elif abs(final_state[0]) >= 2.0:
        termination = 'cart_boundary'
    else:
        termination = 'angle_limit'

    return {
        'states': np.array(states),
        'forces': np.array(forces),
        'length': step,
        'survived': not done,
        'termination': termination,
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
        "--horizon",
        type=int,
        default=5,
        help="Random shooting lookahead horizon (default: 5)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of random trajectories to sample (default: 200)",
    )
    parser.add_argument(
        "--cart_penalty_weight",
        type=float,
        default=1.0,
        help="Weight for cart position penalty in cost function (default: 1.0)",
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
    print(f"Horizon: {args.horizon}, Samples: {args.num_samples}")
    print(f"Cart penalty weight: {args.cart_penalty_weight}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Create controller and environment
    controller = PredictiveController(
        model,
        horizon=args.horizon,
        num_samples=args.num_samples,
        cart_penalty_weight=args.cart_penalty_weight,
        device=device,
    )
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
    termination_counts = {'survived': 0, 'angle_limit': 0, 'cart_boundary': 0}
    for i in range(args.num_episodes):
        result = run_episode(
            env, controller, args.max_steps,
            args.angle_range, args.angle_range,
        )
        rpl_lengths.append(result['length'])
        termination_counts[result['termination']] += 1

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

    # Termination cause breakdown
    total = sum(termination_counts.values())
    print(f"\nTermination causes:")
    print(f"  Angle limit:    {termination_counts['angle_limit']:3d} ({100*termination_counts['angle_limit']/total:.1f}%)")
    print(f"  Cart boundary:  {termination_counts['cart_boundary']:3d} ({100*termination_counts['cart_boundary']/total:.1f}%)")
    print(f"  Survived:       {termination_counts['survived']:3d} ({100*termination_counts['survived']/total:.1f}%)")

    if improvement > 2:
        print("\n✓ RPL controller significantly outperforms random control!")
    elif improvement > 1.2:
        print("\n~ RPL controller shows moderate improvement")
    else:
        print("\n✗ RPL controller needs more training to show clear improvement")


if __name__ == "__main__":
    main()
