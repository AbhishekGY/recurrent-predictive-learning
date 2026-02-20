"""
Real-Time Visualization for RPL Inverted Pendulum Control

Displays a three-panel animation during a control episode:
- Left:   Physical pendulum simulation (cart + pole)
- Center: LSTM representation trajectory in PCA space
- Right:  Best trajectory rollout in PCA space (from random shooting)

Usage:
    python -m pendulum.animate --checkpoint checkpoints/rpl_model_final.pt --save pendulum.gif
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from sklearn.decomposition import PCA

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel
from pendulum.control import PredictiveController
from pendulum.evaluate import collect_test_episodes, collect_representations


def prefit_pca(
    model: RPLModel,
    device: torch.device,
    num_episodes: int = 50,
) -> PCA:
    """
    Pre-fit PCA on representations from random episodes so projection
    axes are stable during the animation.
    """
    episodes = collect_test_episodes(num_episodes=num_episodes, seed=99999)
    representations, _ = collect_representations(model, episodes, device)

    pca = PCA(n_components=2)
    pca.fit(representations)
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")
    return pca


class PendulumAnimator:
    """
    Three-panel real-time visualization of RPL control.

    Left:   Cart-pole rendering
    Center: PCA trajectory of LSTM hidden state
    Right:  Best trajectory rollout in PCA space
    """

    def __init__(
        self,
        model: RPLModel,
        env: InvertedPendulum,
        controller: PredictiveController,
        pca: PCA,
        max_steps: int = 500,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.env = env
        self.controller = controller
        self.pca = pca
        self.max_steps = max_steps
        self.device = device

        # Episode state
        self.state = None
        self.step_count = 0
        self.done = False
        self.trajectory_pca = []  # PCA-projected representations

        # Goal representation projected to PCA space
        # Pass goal state through encoder + integrator (with zero action)
        # to get a 64D representation comparable to trajectory points
        with torch.no_grad():
            goal_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
            goal_action = torch.tensor([[0.0]], device=device)
            goal_repr, _ = model.get_representation(goal_state, goal_action)
        self.goal_pca = pca.transform(goal_repr.cpu().numpy())[0]

        self._setup_figure()

    def _setup_figure(self):
        """Create the three-panel figure."""
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('RPL Predictive Control - Inverted Pendulum', fontsize=14)

        # Left panel: Pendulum
        ax = self.axes[0]
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.3, 0.9)
        ax.set_aspect('equal')
        ax.set_title('Pendulum')

        # Track
        ax.plot([-2, 2], [0, 0], 'k-', linewidth=2)
        ax.plot([-2, -2], [-0.05, 0.05], 'k-', linewidth=2)
        ax.plot([2, 2], [-0.05, 0.05], 'k-', linewidth=2)

        # Cart (rectangle)
        self.cart_patch = plt.Rectangle((-0.15, -0.075), 0.3, 0.15,
                                         fill=True, color='#2196F3')
        ax.add_patch(self.cart_patch)

        # Pole (line)
        self.pole_line, = ax.plot([], [], 'k-', linewidth=3)

        # Bob (circle)
        self.bob_circle = plt.Circle((0, 0.5), 0.04, fill=True, color='#F44336')
        ax.add_patch(self.bob_circle)

        # State text
        self.state_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

        # Step counter
        self.step_text = ax.text(
            0.98, 0.98, '', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
        )

        # Center panel: Representation PCA trajectory
        ax = self.axes[1]
        ax.set_title('Representation Trajectory (PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        self.trail_scatter = ax.scatter([], [], c=[], cmap='viridis', s=15, alpha=0.6)
        self.current_point, = ax.plot([], [], 'ro', markersize=10, zorder=5)
        ax.plot(self.goal_pca[0], self.goal_pca[1], '*',
                color='gold', markersize=15, markeredgecolor='black',
                markeredgewidth=1, zorder=6, label='Goal')
        ax.legend(loc='upper right', fontsize=8)

        # Right panel: Best trajectory rollout
        ax = self.axes[2]
        ax.set_title('Best Trajectory (PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        self.traj_line, = ax.plot([], [], 'b-o', markersize=5, linewidth=1.5,
                                   alpha=0.7, label='Best rollout')
        self.traj_start, = ax.plot([], [], 'gs', markersize=10, label='Current')
        ax.plot(self.goal_pca[0], self.goal_pca[1], '*',
                color='gold', markersize=15, markeredgecolor='black',
                markeredgewidth=1, zorder=6, label='Goal')
        ax.legend(loc='upper right', fontsize=8)

        self.fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

    def _get_best_trajectory_pca(self, state: np.ndarray, best_force: float) -> np.ndarray:
        """
        Roll out the best force sequence through the model and return
        PCA-projected hidden states for each step.
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.model.encoder(state_tensor)
            h = self.controller.hidden

            # Use best_force as first step, then zeros for remainder
            # This shows where the best first action leads
            forces = [best_force] + [0.0] * (self.controller.horizon - 1)

            pca_points = []
            for force in forces:
                force_tensor = torch.tensor([[force]], dtype=torch.float32,
                                            device=self.device)
                h_out, h = self.model.integrator(z, force_tensor, h)
                z = self.model.predictor(h_out)
                pca_pt = self.pca.transform(h_out.cpu().numpy())[0]
                pca_points.append(pca_pt)

        return np.array(pca_points)

    def _reset_episode(self):
        """Reset for a new episode."""
        self.state = self.env.reset(angle_range=0.05, omega_range=0.05)
        self.controller.reset()
        self.step_count = 0
        self.done = False
        self.trajectory_pca = []

    def _update_frame(self, frame_num):
        """Update all three panels for one frame."""
        if self.done or self.step_count >= self.max_steps:
            return []

        state = self.state

        # --- Select action using the real controller (random shooting) ---
        best_force = self.controller.select_action(state)

        # --- Get best trajectory rollout for visualization ---
        traj_pca = self._get_best_trajectory_pca(state, best_force)

        # Execute action in the environment
        self.state, self.done = self.env.step(best_force)

        # Update controller's persistent hidden state
        self.controller.update(self.state, best_force)

        # Get 64D representation for trajectory PCA panel
        with torch.no_grad():
            repr_pca = self.pca.transform(
                self.controller.hidden[0].squeeze(0).cpu().numpy()
            )[0]

        self.trajectory_pca.append(repr_pca)
        self.step_count += 1

        # --- Update Left Panel: Pendulum ---
        x, v_x, theta, omega = self.state
        L = self.env.L

        # Cart position
        self.cart_patch.set_x(x - 0.15)

        # Pole endpoint
        pole_x = x + L * np.sin(theta)
        pole_y = L * np.cos(theta)
        self.pole_line.set_data([x, pole_x], [0, pole_y])

        # Bob position
        self.bob_circle.center = (pole_x, pole_y)

        # State text
        self.state_text.set_text(
            f'x    = {x:+.3f} m\n'
            f'v_x  = {v_x:+.3f} m/s\n'
            f'θ    = {theta:+.3f} rad\n'
            f'ω    = {omega:+.3f} rad/s\n'
            f'F    = {best_force:+.1f} N'
        )
        self.step_text.set_text(f'Step {self.step_count}')

        # --- Update Center Panel: Trajectory ---
        if len(self.trajectory_pca) > 0:
            traj = np.array(self.trajectory_pca)
            # Update scatter with color by time
            self.trail_scatter.set_offsets(traj)
            colors = np.linspace(0.2, 1.0, len(traj))
            self.trail_scatter.set_array(colors)
            self.trail_scatter.set_sizes(np.full(len(traj), 15))

            # Current point
            self.current_point.set_data([traj[-1, 0]], [traj[-1, 1]])

            # Auto-scale
            ax = self.axes[1]
            margin = 0.5
            ax.set_xlim(traj[:, 0].min() - margin, traj[:, 0].max() + margin)
            ax.set_ylim(traj[:, 1].min() - margin, traj[:, 1].max() + margin)

        # --- Update Right Panel: Best trajectory rollout ---
        self.traj_line.set_data(traj_pca[:, 0], traj_pca[:, 1])
        if len(self.trajectory_pca) > 0:
            self.traj_start.set_data([self.trajectory_pca[-1][0]],
                                      [self.trajectory_pca[-1][1]])

        # Auto-scale right panel
        ax = self.axes[2]
        all_pts = np.vstack([traj_pca, [self.goal_pca]])
        if len(self.trajectory_pca) > 0:
            all_pts = np.vstack([all_pts, [self.trajectory_pca[-1]]])
        margin = 0.5
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        return []

    def run(self, save_path: str = None) -> int:
        """
        Run the animation.

        Args:
            save_path: If provided, save animation to this path (GIF or MP4).
                      If None, attempts to show interactively.

        Returns:
            Episode length in steps
        """
        self._reset_episode()

        anim = animation.FuncAnimation(
            self.fig, self._update_frame,
            frames=self.max_steps,
            interval=40,  # 25 fps for viewing
            blit=False,
            repeat=False,
        )

        if save_path:
            save_path = Path(save_path)
            print(f"Saving animation to {save_path}...")

            if save_path.suffix == '.gif':
                writer = animation.PillowWriter(fps=25)
            elif save_path.suffix in ('.mp4', '.avi'):
                writer = animation.FFMpegWriter(fps=25)
            else:
                writer = animation.PillowWriter(fps=25)
                save_path = save_path.with_suffix('.gif')

            anim.save(str(save_path), writer=writer)
            print(f"Saved animation ({self.step_count} frames)")
        else:
            plt.show()

        plt.close(self.fig)
        return self.step_count


def main():
    parser = argparse.ArgumentParser(
        description="Real-time visualization of RPL pendulum control"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/rpl_model_final.pt",
    )
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument(
        "--save", type=str, default="plots/pendulum_control.gif",
        help="Save animation to file (GIF or MP4)",
    )
    parser.add_argument("--horizon", type=int, default=5,
                        help="Random shooting lookahead horizon")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of random trajectories to sample")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== RPL Pendulum Animation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max steps: {args.max_steps}")
    print(f"Horizon: {args.horizon}, Samples: {args.num_samples}")
    print(f"Save to: {args.save}")

    # Load model
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Pre-fit PCA
    print("\nPre-fitting PCA on random episodes...")
    pca = prefit_pca(model, device)

    # Create environment and controller
    env = InvertedPendulum()
    controller = PredictiveController(
        model, horizon=args.horizon, num_samples=args.num_samples, device=device
    )

    # Create animator
    animator = PendulumAnimator(
        model=model,
        env=env,
        controller=controller,
        pca=pca,
        max_steps=args.max_steps,
        device=device,
    )

    # Ensure output directory exists
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    # Run animation
    print("\nRunning episode...")
    steps = animator.run(save_path=args.save)
    print(f"\nEpisode lasted {steps} steps ({steps * 0.02:.2f}s)")


if __name__ == "__main__":
    main()
