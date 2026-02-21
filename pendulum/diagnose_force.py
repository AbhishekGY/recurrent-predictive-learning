"""
Diagnostic script to verify force-response physics and controller sign convention.

Usage:
    python -m pendulum.diagnose_force --checkpoint checkpoints/rpl_model_best_cart.pt
"""

import argparse

import numpy as np
import torch

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel
from pendulum.control import PredictiveController


def test_force_response():
    """Verify that forces produce physically correct responses."""
    env = InvertedPendulum()

    print("=== Test 1: Force Response Verification ===")
    print("Starting state: theta=+0.1 rad (tilting right), everything else zero")
    print("Expected: positive force should move cart right (under the bob)")
    print("          negative force should move cart left\n")

    for force in [-9.0, 0.0, 9.0]:
        env.set_state(np.array([0.0, 0.0, 0.1, 0.0]))  # small rightward tilt
        print(f"Applying constant F={force:+.1f}N for 20 steps:")
        print(f"  {'Step':>4}  {'x':>8}  {'v_x':>8}  {'theta':>8}  {'omega':>8}")

        for step in range(20):
            state, done = env.step(force)
            if step % 5 == 4:  # print every 5 steps
                print(f"  {step+1:>4}  {state[0]:>8.4f}  {state[1]:>8.4f}  "
                      f"{state[2]:>8.4f}  {state[3]:>8.4f}")
            if done:
                print(f"  Episode terminated at step {step+1}")
                break
        print()


def test_controller_sign(model, device):
    """Verify controller selects forces in the correct direction."""
    print("=== Test 2: Controller Sign Convention ===")
    print("For a rightward tilt (theta > 0), controller should select positive force")
    print("For a leftward tilt (theta < 0), controller should select negative force\n")

    controller = PredictiveController(model, horizon=5, num_samples=200, device=device)

    test_states = [
        np.array([0.0, 0.0,  0.2, 0.0], dtype=np.float32),   # rightward tilt
        np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32),   # leftward tilt
        np.array([0.0, 0.0,  0.2, 0.5], dtype=np.float32),   # rightward tilt + falling right
        np.array([0.5, 0.0,  0.1, 0.0], dtype=np.float32),   # cart right + small tilt
        np.array([-0.5, 0.0, -0.1, 0.0], dtype=np.float32),  # cart left + small tilt
    ]

    labels = [
        "theta=+0.2 (right tilt)",
        "theta=-0.2 (left tilt)",
        "theta=+0.2, omega=+0.5 (falling right)",
        "x=+0.5, theta=+0.1 (cart right, small tilt)",
        "x=-0.5, theta=-0.1 (cart left, small tilt)",
    ]

    print(f"  {'State':<45} {'Force':>8}  {'Expected':>10}  {'Correct?'}")
    print(f"  {'-'*45} {'-'*8}  {'-'*10}  {'-'*8}")

    for state, label in zip(test_states, labels):
        controller.reset()
        force = controller.select_action(state)

        # Expected direction based on physics
        theta = state[2]
        expected = "positive" if theta > 0 else "negative"
        correct = "Y" if (theta > 0 and force > 0) or (theta < 0 and force < 0) else "X WRONG"

        print(f"  {label:<45} {force:>+8.2f}  {expected:>10}  {correct}")
    print()


def test_torque_adequacy():
    """
    Compare gravitational torque pulling pendulum down vs
    restoring torque from cart acceleration.
    """
    print("=== Test 3: Torque Adequacy ===")

    env = InvertedPendulum()
    M, m, L, g = env.M, env.m, env.L, env.g

    print(f"System: M={M}kg (cart), m={m}kg (pendulum), L={L}m, g={g}m/s^2\n")
    print(f"  {'theta (rad)':>12}  {'Grav torque':>12}  {'9N cart torque':>15}  {'Adequate?':>10}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*15}  {'-'*10}")

    for theta in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        # Gravitational torque pulling pendulum away from upright
        tau_grav = m * g * L * np.sin(theta)

        # Cart acceleration from F=9N (simplified)
        a_cart = 9.0 / (M + m)

        # Restoring torque from cart acceleration (inertial force on pendulum)
        tau_restore = m * a_cart * L * np.cos(theta)

        adequate = "Yes" if tau_restore > tau_grav else "No"
        print(f"  {theta:>12.2f}  {tau_grav:>12.4f}  {tau_restore:>15.4f}  {adequate:>10}")

    print(f"\nNote: tau_restore > tau_grav means 9N is sufficient to arrest the fall at that angle")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose force-response physics and controller sign convention"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/rpl_model_best_cart.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Test 1: No model needed
    test_force_response()

    # Load model for test 2
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_controller_sign(model, device)
    test_torque_adequacy()


if __name__ == "__main__":
    main()
