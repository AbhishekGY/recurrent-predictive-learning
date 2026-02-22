"""
Torque Diagnostic Visualization

Runs a single control episode and plots:
1. Pendulum angle (theta) over time
2. Applied force over time
3. Gravitational torque vs restoring torque over time
4. Torque deficit (grav - restore) — positive means controller is losing

Usage:
    python -m pendulum.diagnose_torque --checkpoint checkpoints/rpl_model_best_cart.pt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel
from pendulum.control import PredictiveController


def run_diagnostic_episode(
    model: RPLModel,
    device: torch.device,
    max_steps: int = 500,
    seed: int = 42,
    cart_penalty_weight: float = 0.04,
) -> dict:
    """Run one control episode and collect per-step diagnostic data."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = InvertedPendulum()
    controller = PredictiveController(
        model, horizon=5, num_samples=200,
        cart_penalty_weight=cart_penalty_weight, device=device,
    )

    M, m, L, g = env.M, env.m, env.L, env.g

    state = env.reset(angle_range=0.05, omega_range=0.05)
    controller.reset()

    timesteps = []
    thetas = []
    omegas = []
    forces = []
    grav_torques = []
    restore_torques = []
    cart_positions = []

    done = False
    step = 0

    while not done and step < max_steps:
        force = controller.select_action(state)

        x, vx, theta, omega = state

        # Gravitational torque (pulling pendulum away from upright)
        tau_grav = m * g * L * np.sin(abs(theta))

        # Cart acceleration from applied force (simplified)
        a_cart = force / (M + m)

        # Restoring torque from cart acceleration
        # Sign: restoring when cart accelerates in same direction as tilt
        tau_restore = m * abs(a_cart) * L * np.cos(theta)

        timesteps.append(step)
        thetas.append(theta)
        omegas.append(omega)
        forces.append(force)
        grav_torques.append(tau_grav)
        restore_torques.append(tau_restore)
        cart_positions.append(x)

        state, done = env.step(force)
        controller.update(state, force)
        step += 1

    print(f"Episode lasted {step} steps ({step * 0.02:.2f}s)")
    print(f"Termination: {'angle limit' if abs(state[2]) > np.pi/2 else 'cart boundary'}")

    return {
        'timesteps': np.array(timesteps),
        'thetas': np.array(thetas),
        'omegas': np.array(omegas),
        'forces': np.array(forces),
        'grav_torques': np.array(grav_torques),
        'restore_torques': np.array(restore_torques),
        'cart_positions': np.array(cart_positions),
    }


def plot_diagnostics(data: dict, output_path: Path) -> None:
    """Plot 4-panel diagnostic figure."""
    t = data['timesteps'] * 0.02  # convert to seconds

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Control Episode Torque Diagnostics', fontsize=14)

    # Panel 1: Theta and cart position
    ax = axes[0]
    ax.plot(t, np.degrees(data['thetas']), 'b-', linewidth=1.5, label='theta (degrees)')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.axhline(y=90, color='r', linewidth=1, linestyle='--', alpha=0.5, label='Failure threshold')
    ax.axhline(y=-90, color='r', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_ylabel('Angle (degrees)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add cart position on secondary axis
    ax2 = ax.twinx()
    ax2.plot(t, data['cart_positions'], 'g-', linewidth=1, alpha=0.6, label='x (m)')
    ax2.axhline(y=2.0, color='g', linewidth=1, linestyle=':', alpha=0.5)
    ax2.axhline(y=-2.0, color='g', linewidth=1, linestyle=':', alpha=0.5)
    ax2.set_ylabel('Cart position (m)', color='g')
    ax2.legend(loc='upper left', fontsize=8)

    # Panel 2: Applied force
    ax = axes[1]
    ax.plot(t, data['forces'], 'purple', linewidth=1.5)
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.axhline(y=10, color='r', linewidth=1, linestyle='--', alpha=0.4, label='Force limit')
    ax.axhline(y=-10, color='r', linewidth=1, linestyle='--', alpha=0.4)
    ax.set_ylabel('Applied Force (N)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Gravitational vs restoring torque
    ax = axes[2]
    ax.plot(t, data['grav_torques'], 'r-', linewidth=1.5, label='Gravitational torque')
    ax.plot(t, data['restore_torques'], 'b-', linewidth=1.5, label='Restoring torque (from cart)')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Torque (N*m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Torque deficit (positive = controller losing)
    ax = axes[3]
    deficit = data['grav_torques'] - data['restore_torques']
    ax.fill_between(t, deficit, 0,
                    where=(deficit > 0), color='red', alpha=0.4,
                    label='Losing (grav > restore)')
    ax.fill_between(t, deficit, 0,
                    where=(deficit <= 0), color='green', alpha=0.4,
                    label='Winning (restore > grav)')
    ax.plot(t, deficit, 'k-', linewidth=1)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_ylabel('Torque deficit (N*m)')
    ax.set_xlabel('Time (seconds)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Torque diagnostic visualization for control episodes"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/rpl_model_best_cart.pt")
    parser.add_argument("--output", type=str,
                        default="plots/torque_diagnostics.png")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cart_penalty_weight", type=float, default=0.04)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Running diagnostic episode...")
    data = run_diagnostic_episode(model, device, args.max_steps, args.seed,
                                   args.cart_penalty_weight)

    plot_diagnostics(data, Path(args.output))


if __name__ == "__main__":
    main()
