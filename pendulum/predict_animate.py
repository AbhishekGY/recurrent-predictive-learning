"""
Multi-Step Prediction Visualization for Passive RPL Model

Shows real pendulum evolving under passive physics (zero force) alongside
autoregressive N-step predicted trajectories decoded from the model's
embedding space using linear decoders.

Three-panel animation:
  Left:   Cart-pendulum rendering (real + ghost predictions)
  Center: Phase portrait (theta vs omega)
  Right:  Time series of theta with prediction horizon

Usage:
    python -m pendulum.predict_animate --checkpoint checkpoints/rpl_model_final.pt
    python -m pendulum.predict_animate --checkpoint checkpoints/rpl_model_final.pt --n_steps 15
    python -m pendulum.predict_animate --n_steps 10 --save plots/prediction_animation.gif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel
from pendulum.evaluate import collect_test_episodes, collect_representations


def train_embedding_decoders(
    model: RPLModel,
    device: torch.device,
    num_episodes: int = 200,
    seed: int = 12345,
) -> list:
    """
    Train linear decoders that map predicted embeddings (32D) to state [x, v_x, theta, omega].

    Collects representations from held-out passive episodes, then trains
    one LinearRegression per state variable on the encoder output space.

    Returns:
        List of 4 LinearRegression models (one per state variable).
    """
    print("Training linear decoders from embedding space...")
    episodes = collect_test_episodes(
        num_episodes=num_episodes,
        max_steps=500,
        theta_range=0.8,
        omega_range=0.8,
        x_range=0.5,
        vx_range=0.3,
        seed=seed,
    )

    # Collect embeddings (encoder outputs) paired with next states
    all_embeddings = []
    all_next_states = []

    with torch.no_grad():
        for episode in episodes:
            states = episode['states']
            T = len(states) - 1
            for t in range(T):
                state_t = torch.tensor(states[t + 1:t + 2], dtype=torch.float32, device=device)
                z = model.encoder(state_t)
                all_embeddings.append(z.cpu().numpy().flatten())
                all_next_states.append(states[t + 1])

    embeddings = np.array(all_embeddings)
    next_states = np.array(all_next_states)

    state_names = ['x (position)', 'v_x (velocity)', 'theta (angle)', 'omega (ang. vel.)']
    decoders = []

    print(f"  Trained on {len(embeddings):,} samples")
    for i, name in enumerate(state_names):
        dec = LinearRegression()
        dec.fit(embeddings, next_states[:, i])
        preds = dec.predict(embeddings)
        ss_res = np.sum((next_states[:, i] - preds) ** 2)
        ss_tot = np.sum((next_states[:, i] - next_states[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  {name}: R² = {r2:.4f}")
        decoders.append(dec)

    return decoders


def run_real_episode(
    max_steps: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Run a passive episode and return all states.

    Returns:
        real_states: np.array of shape (T, 4)
    """
    np.random.seed(seed)
    env = InvertedPendulum()
    env.reset(
        angle_range=0.3,
        omega_range=0.3,
        x_range=0.3,
        vx_range=0.2,
    )

    states = [env.get_state().copy()]
    for _ in range(max_steps):
        state, done = env.step(0.0)
        states.append(state.copy())
        if done:
            break

    return np.array(states, dtype=np.float32)


def precompute_hidden_states(
    model: RPLModel,
    real_states: np.ndarray,
    device: torch.device,
) -> list:
    """
    Run the model forward through the full real trajectory and store
    the LSTM hidden state at each timestep.

    Returns:
        hidden_states: list of (h, c) tuples, length T.
            hidden_states[t] is the hidden state AFTER processing real_states[t].
    """
    hidden_states = []
    hidden = None

    with torch.no_grad():
        for t in range(len(real_states)):
            state_t = torch.tensor(real_states[t:t + 1], dtype=torch.float32, device=device)
            _, _, hidden = model.forward_step(state_t, hidden)
            # Clone to detach from the computation graph
            hidden_states.append((hidden[0].clone(), hidden[1].clone()))

    return hidden_states


def predict_ahead(
    model: RPLModel,
    z_current: torch.Tensor,
    hidden: tuple,
    n_steps: int,
    decoders: list,
    device: torch.device,
) -> np.ndarray:
    """
    From current embedding z_current and hidden state,
    autoregressively predict n_steps ahead.

    Returns:
        predicted_states: np.array of shape (n_steps, 4)
            decoded from predicted embeddings using linear decoders
    """
    predicted_states = []

    # Work on copies so we don't modify the originals
    z = z_current.clone()
    h = (hidden[0].clone(), hidden[1].clone())

    with torch.no_grad():
        for _ in range(n_steps):
            h_out, h = model.integrator(z, h)
            z = model.predictor(h_out)

            # Decode embedding to state via linear decoders
            z_np = z.cpu().numpy().flatten()
            decoded = np.array([dec.predict(z_np.reshape(1, -1))[0] for dec in decoders])
            predicted_states.append(decoded)

    return np.array(predicted_states)


def create_animation(
    model: RPLModel,
    real_states: np.ndarray,
    hidden_states: list,
    decoders: list,
    n_steps: int,
    device: torch.device,
    save_path: str,
    no_save: bool,
    pendulum_length: float = 0.5,
):
    """Build and save/show the three-panel animation."""
    import matplotlib
    if not no_save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    T = len(real_states)

    # Pre-compute all predicted trajectories
    print("Pre-computing prediction rollouts...")
    all_predictions = []
    with torch.no_grad():
        for t in range(T):
            state_t = torch.tensor(real_states[t:t + 1], dtype=torch.float32, device=device)
            z_current = model.encoder(state_t)
            remaining = min(n_steps, T - 1 - t)
            if remaining > 0:
                pred = predict_ahead(model, z_current, hidden_states[t], remaining, decoders, device)
            else:
                pred = np.empty((0, 4))
            all_predictions.append(pred)
    print(f"  Pre-computed {T} rollouts")

    # --- Figure setup ---
    fig, (ax_pend, ax_phase, ax_series) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('RPL Multi-Step Prediction Visualization', fontsize=14)

    L = pendulum_length

    # --- Panel 1: Pendulum ---
    ax_pend.set_xlim(-2.5, 2.5)
    ax_pend.set_ylim(-0.7, 0.9)
    ax_pend.set_aspect('equal')
    ax_pend.set_title('Pendulum (real + predicted)')

    # Track
    ax_pend.plot([-2, 2], [0, 0], 'k-', linewidth=2)
    ax_pend.axvline(-2, ymin=0.42, ymax=0.58, color='r', linewidth=2)
    ax_pend.axvline(2, ymin=0.42, ymax=0.58, color='r', linewidth=2)

    # Real cart
    cart_patch = plt.Rectangle((-0.15, -0.075), 0.3, 0.15, fill=True, color='#2196F3')
    ax_pend.add_patch(cart_patch)

    # Real pole + bob
    pole_line, = ax_pend.plot([], [], 'k-', linewidth=3)
    bob_circle = plt.Circle((0, L), 0.04, fill=True, color='#333333')
    ax_pend.add_patch(bob_circle)

    # Ghost elements (pre-create for max n_steps)
    ghost_lines = []
    ghost_bobs = []
    for i in range(n_steps):
        alpha = 0.7 - 0.6 * (i / max(n_steps - 1, 1))
        radius = 0.035 - 0.02 * (i / max(n_steps - 1, 1))
        gl, = ax_pend.plot([], [], '-', color='#FF5722', linewidth=2, alpha=alpha)
        gb = plt.Circle((0, 0), radius, fill=True, color='#FF5722', alpha=alpha)
        ax_pend.add_patch(gb)
        ghost_lines.append(gl)
        ghost_bobs.append(gb)

    # State text
    state_text = ax_pend.text(
        0.02, 0.98, '', transform=ax_pend.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    # --- Panel 2: Phase portrait ---
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel(r'$\theta$ (rad)')
    ax_phase.set_ylabel(r'$\omega$ (rad/s)')

    # Full real trajectory (faint)
    ax_phase.plot(real_states[:, 2], real_states[:, 3], '-', color='lightgray', linewidth=0.8)
    # Crosshairs
    ax_phase.axhline(0, color='lightgray', linewidth=0.5, linestyle='--')
    ax_phase.axvline(0, color='lightgray', linewidth=0.5, linestyle='--')

    phase_real_line, = ax_phase.plot([], [], '-', color='#2196F3', linewidth=1.5)
    phase_real_dot, = ax_phase.plot([], [], 'o', color='#2196F3', markersize=8, zorder=5)
    phase_pred_line, = ax_phase.plot([], [], '--', color='#FF5722', linewidth=1.5)
    phase_pred_dots, = ax_phase.plot([], [], 'o', color='#FF5722', markersize=4, zorder=4)

    # Auto-range for phase
    theta_all = real_states[:, 2]
    omega_all = real_states[:, 3]
    th_margin = max(0.3, (theta_all.max() - theta_all.min()) * 0.15)
    om_margin = max(0.3, (omega_all.max() - omega_all.min()) * 0.15)
    ax_phase.set_xlim(theta_all.min() - th_margin, theta_all.max() + th_margin)
    ax_phase.set_ylim(omega_all.min() - om_margin, omega_all.max() + om_margin)

    # --- Panel 3: Time series ---
    ax_series.set_title(r'$\theta$ Time Series')
    ax_series.set_xlabel('Timestep')
    ax_series.set_ylabel(r'$\theta$ (rad)')
    ax_series.axhline(np.pi / 2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_series.axhline(-np.pi / 2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    series_real_line, = ax_series.plot([], [], '-', color='#2196F3', linewidth=1.5)
    series_pred_line, = ax_series.plot([], [], '--', color='#FF5722', linewidth=1.5)
    series_vline = ax_series.axvline(0, color='gray', linewidth=1, linestyle='-', alpha=0.5)

    window = 80

    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.30, top=0.90, bottom=0.12)

    def update(frame):
        real = real_states[frame]
        x, v_x, theta, omega = real
        pred = all_predictions[frame]

        # --- Panel 1: Pendulum ---
        cart_patch.set_x(x - 0.15)
        pole_x = x + L * np.sin(theta)
        pole_y = L * np.cos(theta)
        pole_line.set_data([x, pole_x], [0, pole_y])
        bob_circle.center = (pole_x, pole_y)

        # Ghost pendulums
        for i in range(n_steps):
            if i < len(pred):
                px, _, ptheta, _ = pred[i]
                gx = px + L * np.sin(ptheta)
                gy = L * np.cos(ptheta)
                ghost_lines[i].set_data([px, gx], [0, gy])
                ghost_bobs[i].center = (gx, gy)
                ghost_lines[i].set_visible(True)
                ghost_bobs[i].set_visible(True)
            else:
                ghost_lines[i].set_visible(False)
                ghost_bobs[i].set_visible(False)

        state_text.set_text(
            f'Step {frame}\n'
            f'x   = {x:+.3f} m\n'
            f'v_x = {v_x:+.3f} m/s\n'
            f'\u03b8   = {theta:+.3f} rad\n'
            f'\u03c9   = {omega:+.3f} rad/s'
        )

        # --- Panel 2: Phase portrait ---
        t_start = max(0, frame - 200)
        phase_real_line.set_data(real_states[t_start:frame + 1, 2],
                                 real_states[t_start:frame + 1, 3])
        phase_real_dot.set_data([theta], [omega])

        if len(pred) > 0:
            phase_pred_line.set_data(pred[:, 2], pred[:, 3])
            phase_pred_dots.set_data(pred[:, 2], pred[:, 3])
        else:
            phase_pred_line.set_data([], [])
            phase_pred_dots.set_data([], [])

        # --- Panel 3: Time series ---
        w_start = max(0, frame - window)
        w_end = frame + 1
        ts_real = np.arange(w_start, w_end)
        series_real_line.set_data(ts_real, real_states[w_start:w_end, 2])

        if len(pred) > 0:
            ts_pred = np.arange(frame, frame + len(pred))
            series_pred_line.set_data(ts_pred, pred[:, 2])
        else:
            series_pred_line.set_data([], [])

        series_vline.set_xdata([frame])

        # Rolling window x-limits
        view_start = max(0, frame - window)
        view_end = frame + n_steps + 5
        ax_series.set_xlim(view_start, view_end)

        # y-limits based on visible data
        vis_theta = real_states[view_start:min(w_end, T), 2]
        if len(pred) > 0:
            vis_theta = np.concatenate([vis_theta, pred[:, 2]])
        if len(vis_theta) > 0:
            y_margin = max(0.1, (vis_theta.max() - vis_theta.min()) * 0.15)
            ax_series.set_ylim(vis_theta.min() - y_margin, vis_theta.max() + y_margin)

        return []

    anim = animation.FuncAnimation(
        fig, update,
        frames=T,
        interval=40,
        blit=False,
        repeat=False,
    )

    if no_save:
        plt.show()
    else:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving animation to {save_p} ({T} frames)...")

        if save_p.suffix == '.mp4':
            writer = animation.FFMpegWriter(fps=25)
        else:
            writer = animation.PillowWriter(fps=25)

        anim.save(str(save_p), writer=writer,
                  progress_callback=lambda i, n: print(f"\r  Frame {i+1}/{n}", end='', flush=True))
        print(f"\n  Saved: {save_p}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-step prediction animation for passive RPL model"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/rpl_model_final.pt",
    )
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Prediction horizon (default: 10)")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Real episode length (default: 300)")
    parser.add_argument("--save", type=str, default="plots/prediction_animation.gif",
                        help="Output path (default: plots/prediction_animation.gif)")
    parser.add_argument("--no_save", action="store_true",
                        help="Show interactively instead of saving")
    parser.add_argument("--num_decoder_episodes", type=int, default=200,
                        help="Episodes for training linear decoders (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for real episode initial condition (default: 42)")
    parser.add_argument("--decoder_seed", type=int, default=12345,
                        help="Seed for decoder training episodes (default: 12345)")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== RPL Multi-Step Prediction Animation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prediction horizon: {args.n_steps} steps")
    print(f"Max episode length: {args.max_steps}")
    print(f"Device: {device}")
    print()

    # Load model
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print()

    # Step 1: Train embedding decoders
    decoders = train_embedding_decoders(
        model, device,
        num_episodes=args.num_decoder_episodes,
        seed=args.decoder_seed,
    )
    print()

    # Step 2: Run real episode
    print("Running real episode (passive dynamics)...")
    real_states = run_real_episode(max_steps=args.max_steps, seed=args.seed)
    print(f"  Episode length: {len(real_states)} timesteps")
    print()

    # Pre-compute hidden states
    print("Pre-computing LSTM hidden states...")
    hidden_states = precompute_hidden_states(model, real_states, device)
    print(f"  Stored {len(hidden_states)} hidden states")
    print()

    # Step 3-6: Create animation
    env = InvertedPendulum()
    create_animation(
        model=model,
        real_states=real_states,
        hidden_states=hidden_states,
        decoders=decoders,
        n_steps=args.n_steps,
        device=device,
        save_path=args.save,
        no_save=args.no_save,
        pendulum_length=env.L,
    )


if __name__ == "__main__":
    main()
