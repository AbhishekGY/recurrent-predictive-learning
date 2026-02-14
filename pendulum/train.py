"""
Training Script for RPL Model

Loads collected episodes and trains the RPL model to predict
next state embeddings from current state-action pairs.

Usage:
    python -m pendulum.train --data_path data/random_exploration.pkl --epochs 100
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pendulum.model import RPLModel, compute_prediction_loss


class EpisodeDataset(Dataset):
    """
    PyTorch Dataset for RPL training episodes.

    Handles padding/truncating episodes to fixed sequence length and
    provides masks for ignoring padded timesteps in loss computation.
    """

    def __init__(self, episodes: list, seq_len: int = 50):
        """
        Initialize the dataset.

        Args:
            episodes: List of episode dicts with 'states' and 'forces' keys
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
                - actions: Tensor of shape (seq_len, 1)
                - mask: Boolean tensor of shape (seq_len,) indicating valid timesteps
        """
        episode = self.episodes[idx]
        states = episode['states']  # (T+1, 4)
        forces = episode['forces']  # (T, 1)

        T = len(forces)  # Number of transitions

        if T >= self.seq_len:
            # Episode is long enough - randomly sample a contiguous window
            start_idx = np.random.randint(0, T - self.seq_len + 1)
            end_idx = start_idx + self.seq_len

            # States: need seq_len+1 states for seq_len transitions
            states_out = states[start_idx:end_idx + 1]  # (seq_len+1, 4)
            actions_out = forces[start_idx:end_idx]     # (seq_len, 1)
            mask = np.ones(self.seq_len, dtype=np.float32)

        else:
            # Episode is shorter - pad with zeros
            states_out = np.zeros((self.seq_len + 1, 4), dtype=np.float32)
            actions_out = np.zeros((self.seq_len, 1), dtype=np.float32)
            mask = np.zeros(self.seq_len, dtype=np.float32)

            # Fill in actual data
            states_out[:T + 1] = states
            actions_out[:T] = forces
            mask[:T] = 1.0

        return (
            torch.from_numpy(states_out),
            torch.from_numpy(actions_out),
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
) -> float:
    """
    Train for one epoch.

    Args:
        model: The RPL model
        dataloader: Training data loader
        optimizer: The optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Mean loss over the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for states, actions, mask in dataloader:
        # Move to device
        states = states.to(device)
        actions = actions.to(device)
        mask = mask.to(device)

        # Forward pass
        output = model(states, actions)

        # Compute loss with masking
        loss = compute_prediction_loss(output['predictions'], output['targets'], mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: RPLModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    epoch_losses: list = None,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if epoch_losses is not None:
        checkpoint['epoch_losses'] = epoch_losses
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train RPL model on collected episodes")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/random_exploration.pkl",
        help="Path to the training data (default: data/random_exploration.pkl)",
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

    print("=== RPL Training ===")
    print(f"Data path: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Learning rates: encoder/integrator={args.lr_slow}, predictor={args.lr_fast}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes")

    # Create dataset and dataloader
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
    model = RPLModel()
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
        loss = train_epoch(model, dataloader, optimizer, device)
        losses.append(loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {loss:.6f}")

        # Track best loss
        if loss < best_loss:
            best_loss = loss

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            checkpoint_path = checkpoint_dir / f"rpl_model_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_path,
                           epoch_losses=losses)
            print(f"  -> Saved checkpoint: {checkpoint_path}")

    print("-" * 50)
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

    # Save final model
    final_path = checkpoint_dir / "rpl_model_final.pt"
    save_checkpoint(model, optimizer, args.epochs, losses[-1], final_path,
                    epoch_losses=losses)
    print(f"Final model saved to: {final_path}")

    # Print loss progression
    print("\nLoss progression (every 10 epochs):")
    for i in range(0, len(losses), 10):
        print(f"  Epoch {i+1:3d}: {losses[i]:.6f}")


if __name__ == "__main__":
    main()
