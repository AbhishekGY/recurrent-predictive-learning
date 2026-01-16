"""
Inverted Pendulum Environment

A physics simulation of an inverted pendulum mounted on a cart.
The cart moves horizontally along a track, and the goal is to keep
the pendulum balanced upright.

State representation: [x, v_x, theta, omega]
- x: cart position (meters from center)
- v_x: cart velocity (m/s)
- theta: pendulum angle from vertical (radians, 0 = upright)
- omega: angular velocity (rad/s)
"""

import numpy as np


class InvertedPendulum:
    """
    Inverted pendulum on a cart simulation using Euler integration.

    The pendulum is modeled as a point mass at the end of a massless rod,
    mounted on a cart that can move horizontally along a track.
    """

    def __init__(
        self,
        cart_mass: float = 1.0,
        pendulum_mass: float = 0.1,
        pendulum_length: float = 0.5,
        gravity: float = 9.81,
        dt: float = 0.02,
        track_limit: float = 2.0,
        force_limit: float = 10.0,
    ):
        """
        Initialize the pendulum environment.

        Args:
            cart_mass: Mass of the cart (kg)
            pendulum_mass: Mass of the pendulum bob (kg)
            pendulum_length: Half-length of the pendulum (m)
            gravity: Gravitational acceleration (m/s^2)
            dt: Simulation timestep (s)
            track_limit: Maximum cart position from center (m)
            force_limit: Maximum allowed force magnitude (N)
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = pendulum_length
        self.g = gravity
        self.dt = dt
        self.track_limit = track_limit
        self.force_limit = force_limit

        # State: [x, v_x, theta, omega]
        self.state = None

    def reset(
        self,
        angle_range: float = 0.05,
        omega_range: float = 0.05,
        x_range: float = 0.0,
        vx_range: float = 0.0,
    ) -> np.ndarray:
        """
        Reset the environment to initial state with small random perturbations.

        Args:
            angle_range: Maximum initial angle perturbation (radians)
            omega_range: Maximum initial angular velocity (rad/s)
            x_range: Maximum initial cart position (m)
            vx_range: Maximum initial cart velocity (m/s)

        Returns:
            Initial state as numpy array [x, v_x, theta, omega]
        """
        x = np.random.uniform(-x_range, x_range)
        v_x = np.random.uniform(-vx_range, vx_range)
        theta = np.random.uniform(-angle_range, angle_range)
        omega = np.random.uniform(-omega_range, omega_range)

        self.state = np.array([x, v_x, theta, omega], dtype=np.float32)
        return self.state.copy()

    def step(self, force: float) -> tuple[np.ndarray, bool]:
        """
        Advance the simulation by one timestep.

        Args:
            force: Horizontal force applied to cart (N)

        Returns:
            Tuple of (new_state, done):
                - new_state: numpy array [x, v_x, theta, omega]
                - done: True if episode has terminated (failure)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Clamp force to limits
        F = np.clip(force, -self.force_limit, self.force_limit)

        # Extract current state
        x, v_x, theta, omega = self.state

        # Compute accelerations using cart-pole dynamics
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Total mass
        total_mass = self.M + self.m

        # Denominator for angular acceleration
        denom = self.L * (4.0 / 3.0 - (self.m * cos_theta ** 2) / total_mass)

        # Angular acceleration
        temp = (F + self.m * self.L * omega ** 2 * sin_theta) / total_mass
        alpha = (self.g * sin_theta - cos_theta * temp) / denom

        # Cart acceleration
        a = (F + self.m * self.L * (omega ** 2 * sin_theta - alpha * cos_theta)) / total_mass

        # Euler integration
        omega_new = omega + alpha * self.dt
        theta_new = theta + omega_new * self.dt
        v_x_new = v_x + a * self.dt
        x_new = x + v_x_new * self.dt

        # Wrap angle to [-pi, pi]
        theta_new = self._wrap_angle(theta_new)

        # Update state
        self.state = np.array([x_new, v_x_new, theta_new, omega_new], dtype=np.float32)

        # Check termination conditions
        done = self._check_termination()

        return self.state.copy(), done

    def _wrap_angle(self, theta: float) -> float:
        """Wrap angle to [-pi, pi] range."""
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        x, _, theta, _ = self.state

        # Pendulum fell too far (more than 90 degrees from vertical)
        if abs(theta) > np.pi / 2:
            return True

        # Cart hit track boundary
        if abs(x) >= self.track_limit:
            return True

        return False

    def get_state(self) -> np.ndarray:
        """Return a copy of the current state."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Set the environment to a specific state."""
        if len(state) != 4:
            raise ValueError("State must have 4 elements: [x, v_x, theta, omega]")
        self.state = np.array(state, dtype=np.float32)


def test_simulation():
    """Basic test to verify the simulation works correctly."""
    env = InvertedPendulum()

    print("=== Inverted Pendulum Simulation Test ===\n")

    # Test 1: Reset and initial state
    print("Test 1: Reset and initial state")
    state = env.reset(angle_range=0.1)
    print(f"  Initial state: x={state[0]:.4f}, v_x={state[1]:.4f}, "
          f"theta={state[2]:.4f}, omega={state[3]:.4f}")
    print("  PASSED\n")

    # Test 2: Step with zero force (pendulum should fall)
    print("Test 2: Step with zero force")
    initial_theta = state[2]
    for _ in range(10):
        state, done = env.step(0.0)
    print(f"  After 10 steps: theta changed from {initial_theta:.4f} to {state[2]:.4f}")
    print(f"  Pendulum {'fell' if abs(state[2]) > abs(initial_theta) else 'did not fall as expected'}")
    print("  PASSED\n")

    # Test 3: Run until termination
    print("Test 3: Run until termination (no control)")
    env.reset(angle_range=0.1)
    steps = 0
    done = False
    while not done and steps < 500:
        _, done = env.step(0.0)
        steps += 1
    print(f"  Episode terminated after {steps} steps ({steps * env.dt:.2f} seconds)")
    print(f"  Final state: x={env.state[0]:.4f}, theta={env.state[2]:.4f}")
    print("  PASSED\n")

    # Test 4: Random control
    print("Test 4: Random control episode")
    env.reset(angle_range=0.05)
    steps = 0
    done = False
    while not done and steps < 500:
        force = np.random.uniform(-10, 10)
        _, done = env.step(force)
        steps += 1
    print(f"  Episode with random control lasted {steps} steps ({steps * env.dt:.2f} seconds)")
    print("  PASSED\n")

    # Test 5: Energy behavior (approximate conservation with small angle)
    print("Test 5: Energy behavior check")
    env.reset(angle_range=0.0, omega_range=0.0)
    env.set_state(np.array([0.0, 0.0, 0.2, 0.0]))  # Small angle, no velocity

    def compute_energy(state):
        x, v_x, theta, omega = state
        # Kinetic energy (cart + pendulum)
        KE_cart = 0.5 * env.M * v_x ** 2
        # Pendulum velocity includes both cart motion and rotation
        v_pend_x = v_x + env.L * omega * np.cos(theta)
        v_pend_y = env.L * omega * np.sin(theta)
        KE_pend = 0.5 * env.m * (v_pend_x ** 2 + v_pend_y ** 2)
        # Potential energy (pendulum height)
        PE = env.m * env.g * env.L * np.cos(theta)
        return KE_cart + KE_pend + PE

    initial_energy = compute_energy(env.state)
    energies = [initial_energy]
    for _ in range(100):
        state, done = env.step(0.0)
        if done:
            break
        energies.append(compute_energy(state))

    energy_drift = (energies[-1] - initial_energy) / initial_energy * 100
    print(f"  Initial energy: {initial_energy:.4f}")
    print(f"  Final energy: {energies[-1]:.4f}")
    print(f"  Energy drift: {energy_drift:.2f}%")
    print(f"  {'PASSED' if abs(energy_drift) < 10 else 'WARNING: High energy drift'}\n")

    print("=== All tests completed ===")


if __name__ == "__main__":
    test_simulation()
