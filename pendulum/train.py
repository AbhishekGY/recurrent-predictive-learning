"""
Training Script for RPL Model

Loads collected episodes and trains the RPL model to predict
next state embeddings from passive observation sequences.

Usage:
    python -m pendulum.train --data_path data/passive_swings.pkl --epochs 100
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pendulum.model import (
    RPLModel,
    compute_prediction_loss,
    compute_vic_regularization,
)


class ImageEpisodeDataset(Dataset):
    """
    PyTorch Dataset for RPL training from image observations.

    Same windowing/padding logic as EpisodeDataset but operates on
    episode['images'] with shape (T+1, 1, 64, 64).
    """

    def __init__(self, episodes: list, seq_len: int = 50):
        self.episodes = episodes
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            Tuple of:
                - images: Tensor of shape (seq_len+1, 1, 64, 64)
                - mask: Boolean tensor of shape (seq_len,)
        """
        episode = self.episodes[idx]
        images = episode['images']  # (T+1, 1, 64, 64)
        # Support uint8 images saved by collect_image_data.py
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0

        T = len(images) - 1

        if T >= self.seq_len:
            start_idx = np.random.randint(0, T - self.seq_len + 1)
            end_idx = start_idx + self.seq_len
            images_out = images[start_idx:end_idx + 1]
            mask = np.ones(self.seq_len, dtype=np.float32)
        else:
            images_out = np.zeros((self.seq_len + 1, 1, 64, 64), dtype=np.float32)
            mask = np.zeros(self.seq_len, dtype=np.float32)
            images_out[:T + 1] = images
            mask[:T] = 1.0

        return (
            torch.from_numpy(images_out),
            torch.from_numpy(mask),
        )


class EpisodeDataset(Dataset):
    """
    PyTorch Dataset for RPL training episodes (passive observation).

    Handles padding/truncating episodes to fixed sequence length and
    provides masks for ignoring padded timesteps in loss computation.
    """

    def __init__(self, episodes: list, seq_len: int = 50):
        """
        Initialize the dataset.

        Args:
            episodes: List of episode dicts with 'states' key
            seq_len: Fixed sequence length for batching (number of transitions)
        """
        self.episodes = episodes
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single episode, padded or truncated to seq_len.

        Returns:
            Tuple of:
                - states: Tensor of shape (seq_len+1, 4)
                - mask: Boolean tensor of shape (seq_len,) indicating valid timesteps
        """
        episode = self.episodes[idx]
        states = episode['states']  # (T+1, 4)

        T = len(states) - 1  # Number of transitions

        if T >= self.seq_len:
            # Episode is long enough - randomly sample a contiguous window
            start_idx = np.random.randint(0, T - self.seq_len + 1)
            end_idx = start_idx + self.seq_len

            # States: need seq_len+1 states for seq_len transitions
            states_out = states[start_idx:end_idx + 1]  # (seq_len+1, 4)
            mask = np.ones(self.seq_len, dtype=np.float32)

        else:
            # Episode is shorter - pad with zeros
            states_out = np.zeros((self.seq_len + 1, 4), dtype=np.float32)
            mask = np.zeros(self.seq_len, dtype=np.float32)

            # Fill in actual data
            states_out[:T + 1] = states
            mask[:T] = 1.0

        return (
            torch.from_numpy(states_out),
            torch.from_numpy(mask),
        )


def create_optimizer(model: RPLModel, lr_slow: float, lr_fast: float) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with separate learning rates for different components.

    Args:
        model: The RPL model
        lr_slow: Learning rate for encoder and integrator (typically 3e-4)
        lr_fast: Learning rate for predictor (typically 3e-3, 10x higher)

    Returns:
        AdamW optimizer with two param groups
    """
    param_groups = [
        {
            'params': list(model.encoder.parameters()) + list(model.integrator.parameters()),
            'lr': lr_slow,
        },
        {
            'params': list(model.predictor.parameters()),
            'lr': lr_fast,
        },
    ]
    return torch.optim.AdamW(param_groups)


def train_epoch(
    model: RPLModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    var_coef: float = 0.0,
    cov_coef: float = 0.0,
    vic_gamma: float = 1.0,
) -> dict:
    """
    Train for one epoch.

    Args:
        model: The RPL model
        dataloader: Training data loader
        optimizer: The optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        var_coef: Weight of the VICReg variance term (0 disables it)
        cov_coef: Weight of the VICReg covariance term (0 disables it)
        vic_gamma: Target minimum per-dimension std for the variance term

    Returns:
        Dict of mean epoch losses: 'total', 'invariance', 'variance',
        'covariance'.
    """
    model.train()
    use_vic = var_coef > 0.0 or cov_coef > 0.0
    totals = {'total': 0.0, 'invariance': 0.0, 'variance': 0.0, 'covariance': 0.0}
    num_batches = 0

    for states, mask in dataloader:
        # Move to device
        states = states.to(device)
        mask = mask.to(device)

        # Forward pass
        output = model(states)

        # Invariance term: masked prediction MSE.
        invariance = compute_prediction_loss(output['predictions'], output['targets'], mask)
        loss = invariance

        # Anti-collapse regularization on the encoder embeddings. Use the
        # embeddings that feed the predictor (states[:-1]) so they align with
        # the transition mask.
        if use_vic:
            var_loss, cov_loss = compute_vic_regularization(
                output['embeddings'][:, :-1, :], mask, gamma=vic_gamma
            )
            loss = loss + var_coef * var_loss + cov_coef * cov_loss
        else:
            var_loss = torch.zeros((), device=device)
            cov_loss = torch.zeros((), device=device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Update weights
        optimizer.step()

        totals['total'] += loss.item()
        totals['invariance'] += invariance.item()
        totals['variance'] += var_loss.item()
        totals['covariance'] += cov_loss.item()
        num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def save_checkpoint(
    model: RPLModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    epoch_losses: list = None,
    encoder_type: str = 'mlp',
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'encoder_type': encoder_type,
        'normalize_embeddings': getattr(model, 'normalize_embeddings', True),
    }
    if epoch_losses is not None:
        checkpoint['epoch_losses'] = epoch_losses
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train RPL model on collected episodes")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/passive_swings.pkl",
        help="Path to the training data (default: data/passive_swings.pkl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="Sequence length for training (default: 50)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--lr_slow",
        type=float,
        default=3e-4,
        help="Learning rate for encoder/integrator (default: 3e-4)",
    )
    parser.add_argument(
        "--lr_fast",
        type=float,
        default=3e-3,
        help="Learning rate for predictor (default: 3e-3)",
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
        help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Train from image observations instead of state vectors",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable L2 normalization of embeddings. Kept for ablation; note "
             "the default regime uses VICReg regularization (see --var_coef), "
             "which controls the embedding scale itself and turns L2 "
             "normalization off automatically.",
    )
    parser.add_argument(
        "--var_coef",
        type=float,
        default=1.0,
        help="Weight of the VICReg variance term, the anti-collapse "
             "regularizer (default: 1.0). Set to 0 (together with --cov_coef 0) "
             "to fall back to the L2-normalization regime.",
    )
    parser.add_argument(
        "--cov_coef",
        type=float,
        default=1.0,
        help="Weight of the VICReg covariance (decorrelation) term "
             "(default: 1.0). Set to 0 with --var_coef 0 to disable VICReg.",
    )
    parser.add_argument(
        "--vic_gamma",
        type=float,
        default=1.0,
        help="Target minimum per-dimension embedding std for the VICReg "
             "variance hinge (default: 1.0)",
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    encoder_type = 'cnn' if args.image else 'mlp'

    # Scale-control regime. VICReg (variance/covariance regularization) and L2
    # normalization are alternatives -- they pin different quantities and
    # conflict -- so enabling VICReg turns L2 normalization off.
    use_vic = args.var_coef > 0.0 or args.cov_coef > 0.0
    normalize_embeddings = (not args.no_normalize) and not use_vic
    if use_vic:
        regime = f"vicreg (var={args.var_coef}, cov={args.cov_coef}, gamma={args.vic_gamma})"
    elif normalize_embeddings:
        regime = "l2norm"
    else:
        regime = "none (plain MSE)"

    print("=== RPL Training ===")
    print(f"Data path: {args.data_path}")
    print(f"Encoder type: {encoder_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Learning rates: encoder/integrator={args.lr_slow}, predictor={args.lr_fast}")
    print(f"Scale-control regime: {regime}")
    print(f"Normalize embeddings: {normalize_embeddings}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes")

    # Create dataset and dataloader
    if args.image:
        dataset = ImageEpisodeDataset(episodes, seq_len=args.seq_len)
    else:
        dataset = EpisodeDataset(episodes, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for reproducibility
        drop_last=True,
    )
    print(f"Created dataloader with {len(dataloader)} batches per epoch")

    # Create model
    model = RPLModel(use_image=args.image, normalize_embeddings=normalize_embeddings)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")

    # Create optimizer
    optimizer = create_optimizer(model, args.lr_slow, args.lr_fast)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    print("-" * 50)

    best_loss = float('inf')
    losses = []

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model, dataloader, optimizer, device,
            var_coef=args.var_coef, cov_coef=args.cov_coef, vic_gamma=args.vic_gamma,
        )
        loss = metrics['total']
        losses.append(loss)

        if use_vic:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss:.6f} "
                  f"| inv: {metrics['invariance']:.6f} "
                  f"var: {metrics['variance']:.6f} cov: {metrics['covariance']:.6f}")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss:.6f}")

        # Track best loss (selection uses the invariance term so the checkpoint
        # reflects predictive quality, not the regularization penalty).
        selection_loss = metrics['invariance'] if use_vic else loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            save_checkpoint(model, optimizer, epoch, selection_loss,
                    checkpoint_dir / "rpl_model_best.pt",
                    epoch_losses=losses, encoder_type=encoder_type)

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            checkpoint_path = checkpoint_dir / f"rpl_model_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_path,
                           epoch_losses=losses, encoder_type=encoder_type)
            print(f"  -> Saved checkpoint: {checkpoint_path}")

    print("-" * 50)
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

    # Save final model
    final_path = checkpoint_dir / "rpl_model_final.pt"
    save_checkpoint(model, optimizer, args.epochs, losses[-1], final_path,
                    epoch_losses=losses, encoder_type=encoder_type)
    print(f"Final model saved to: {final_path}")

    # Print loss progression
    print("\nLoss progression (every 10 epochs):")
    for i in range(0, len(losses), 10):
        print(f"  Epoch {i+1:3d}: {losses[i]:.6f}")


if __name__ == "__main__":
    main()
