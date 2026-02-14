"""
Offline Visualization and Analysis for RPL Inverted Pendulum

Generates analysis plots from a trained model checkpoint:
1. Representation structure (PCA of LSTM hidden states)
2. Prediction accuracy heatmap (error vs state)
3. Linear decoder weight analysis
4. Learning curves (loss over epochs)

Usage:
    python -m pendulum.visualize --checkpoint checkpoints/rpl_model_final.pt
    python -m pendulum.visualize --checkpoint checkpoints/rpl_model_final.pt --plot representation
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel
from pendulum.evaluate import (
    collect_test_episodes,
    collect_representations,
    train_linear_decoders,
)

STATE_NAMES = ['x (position)', 'v_x (velocity)', 'theta (angle)', 'omega (ang. vel.)']


def collect_prediction_errors_by_state(
    model: RPLModel,
    episodes: list,
    device: torch.device,
) -> dict:
    """
    Collect per-timestep prediction errors alongside the ground truth state.

    Returns:
        dict with:
            'states': np.ndarray (N, 4) - state at time t
            'errors': np.ndarray (N,) - ||z_hat_{t+1} - z_{t+1}||^2
    """
    model.eval()
    all_states = []
    all_errors = []

    with torch.no_grad():
        for episode in episodes:
            states = episode['states']
            forces = episode['forces']
            T = len(forces)
            hidden = None

            for t in range(T):
                state_t = torch.tensor(states[t:t+1], dtype=torch.float32, device=device)
                action_t = torch.tensor(forces[t:t+1], dtype=torch.float32, device=device)
                state_next = torch.tensor(states[t+1:t+2], dtype=torch.float32, device=device)

                # Get prediction
                _, prediction, hidden = model.forward_step(state_t, action_t, hidden)

                # Get actual next embedding
                z_next_actual = model.encoder(state_next)

                # Compute error
                error = ((prediction - z_next_actual) ** 2).mean().item()

                all_states.append(states[t])
                all_errors.append(error)

    return {
        'states': np.array(all_states),
        'errors': np.array(all_errors),
    }


def plot_representation_structure(
    representations: np.ndarray,
    states: np.ndarray,
    output_path: Path,
) -> None:
    """
    Project 64-D LSTM representations onto 2 principal components,
    colored by each state variable in a 2x2 grid.
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(representations)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Representation Structure (PCA)\n"
        f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}",
        fontsize=13,
    )

    for i, (ax, name) in enumerate(zip(axes.flat, STATE_NAMES)):
        sc = ax.scatter(
            projected[:, 0], projected[:, 1],
            c=states[:, i], cmap='coolwarm', alpha=0.3, s=5,
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Colored by {name}')
        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_prediction_heatmap(
    states: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
    n_bins: int = 20,
) -> None:
    """
    2D heatmap of mean prediction error binned by (theta, omega).
    """
    theta = states[:, 2]
    omega = states[:, 3]

    # Create bins
    theta_bins = np.linspace(theta.min(), theta.max(), n_bins + 1)
    omega_bins = np.linspace(omega.min(), omega.max(), n_bins + 1)

    # Bin the errors
    error_sum = np.zeros((n_bins, n_bins))
    error_count = np.zeros((n_bins, n_bins))

    theta_idx = np.clip(np.digitize(theta, theta_bins) - 1, 0, n_bins - 1)
    omega_idx = np.clip(np.digitize(omega, omega_bins) - 1, 0, n_bins - 1)

    for t_i, o_i, err in zip(theta_idx, omega_idx, errors):
        error_sum[o_i, t_i] += err
        error_count[o_i, t_i] += 1

    # Compute mean error per bin (NaN for empty bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_error = np.where(error_count > 0, error_sum / error_count, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('lightgray')

    im = ax.pcolormesh(
        theta_bins, omega_bins, mean_error,
        cmap=cmap, shading='flat',
    )
    ax.set_xlabel('theta (rad)')
    ax.set_ylabel('omega (rad/s)')
    ax.set_title('Prediction Accuracy Heatmap\nMean Embedding MSE by State Region')
    fig.colorbar(im, ax=ax, label='Mean Prediction Error')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_decoder_weights(
    decoder_results: dict,
    output_path: Path,
) -> None:
    """
    Visualize linear decoder weights showing which hidden dimensions
    encode which state variables.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Linear Decoder Weights (LSTM Hidden -> State Variables)', fontsize=13)

    for ax, (name, data) in zip(axes.flat, decoder_results.items()):
        weights = data['decoder'].coef_
        r2 = data['r2']
        dims = np.arange(len(weights))

        colors = ['#2196F3' if w >= 0 else '#F44336' for w in weights]
        ax.bar(dims, weights, color=colors, width=0.8)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel('Weight')
        ax.set_title(f'{name}  (R²={r2:.4f})')
        ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Also create a heatmap of all weights
    heatmap_path = output_path.parent / 'decoder_heatmap.png'
    weight_matrix = np.array([
        decoder_results[name]['decoder'].coef_
        for name in decoder_results
    ])

    # Sort columns by max absolute weight
    max_abs = np.max(np.abs(weight_matrix), axis=0)
    sort_idx = np.argsort(-max_abs)
    weight_matrix_sorted = weight_matrix[:, sort_idx]

    fig, ax = plt.subplots(figsize=(16, 4))
    vmax = np.abs(weight_matrix).max()
    im = ax.imshow(
        weight_matrix_sorted, aspect='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
    )
    ax.set_yticks(range(len(decoder_results)))
    ax.set_yticklabels(list(decoder_results.keys()))
    ax.set_xlabel('Hidden Dimension (sorted by importance)')
    ax.set_title('Decoder Weight Heatmap')
    fig.colorbar(im, ax=ax, label='Weight')

    plt.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {heatmap_path}")


def plot_learning_curves(
    checkpoint: dict,
    output_path: Path,
) -> None:
    """
    Plot training loss over epochs from checkpoint history.
    """
    epoch_losses = checkpoint.get('epoch_losses')
    if epoch_losses is None:
        print("  WARNING: No epoch_losses in checkpoint. "
              "Retrain with updated train.py to get loss history.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(epoch_losses) + 1)
    ax.semilogy(epochs, epoch_losses, 'b-', linewidth=1.5, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Loss (MSE, log scale)')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_all_plots(
    model: RPLModel,
    checkpoint: dict,
    device: torch.device,
    output_dir: Path,
    num_episodes: int = 200,
    seed: int = 12345,
) -> None:
    """Generate all offline analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect test data
    print("Collecting test episodes...")
    episodes = collect_test_episodes(
        num_episodes=num_episodes, seed=seed,
    )
    total = sum(len(ep['forces']) for ep in episodes)
    print(f"  {len(episodes)} episodes, {total:,} transitions")

    # Collect representations
    print("Collecting representations...")
    representations, next_states = collect_representations(model, episodes, device)
    print(f"  Shape: {representations.shape}")

    # Train decoders
    print("Training linear decoders...")
    decoder_results = train_linear_decoders(representations, next_states)
    for name, data in decoder_results.items():
        print(f"  {name}: R²={data['r2']:.4f}")

    # Collect prediction errors
    print("Collecting prediction errors...")
    error_data = collect_prediction_errors_by_state(model, episodes, device)
    print(f"  Mean error: {error_data['errors'].mean():.6f}")

    # Generate plots
    print("\nGenerating plots...")

    plot_representation_structure(
        representations, next_states,
        output_dir / 'representation_pca.png',
    )

    plot_prediction_heatmap(
        error_data['states'], error_data['errors'],
        output_dir / 'prediction_heatmap.png',
    )

    plot_decoder_weights(
        decoder_results,
        output_dir / 'decoder_weights.png',
    )

    plot_learning_curves(
        checkpoint,
        output_dir / 'learning_curves.png',
    )

    print(f"\nAll plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for trained RPL model"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/rpl_model_final.pt",
    )
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument(
        "--plot", type=str, default="all",
        choices=["all", "representation", "prediction", "decoder", "learning"],
    )
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== RPL Visualization ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Plot: {args.plot}")
    print()

    # Load model
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    output_dir = Path(args.output_dir)

    if args.plot == "all":
        generate_all_plots(model, checkpoint, device, output_dir,
                           args.num_episodes, args.seed)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect common data
        episodes = collect_test_episodes(
            num_episodes=args.num_episodes, seed=args.seed,
        )
        representations, next_states = collect_representations(model, episodes, device)

        if args.plot == "representation":
            plot_representation_structure(
                representations, next_states,
                output_dir / 'representation_pca.png',
            )
        elif args.plot == "prediction":
            error_data = collect_prediction_errors_by_state(model, episodes, device)
            plot_prediction_heatmap(
                error_data['states'], error_data['errors'],
                output_dir / 'prediction_heatmap.png',
            )
        elif args.plot == "decoder":
            decoder_results = train_linear_decoders(representations, next_states)
            plot_decoder_weights(
                decoder_results,
                output_dir / 'decoder_weights.png',
            )
        elif args.plot == "learning":
            plot_learning_curves(checkpoint, output_dir / 'learning_curves.png')


if __name__ == "__main__":
    main()
