"""
Predictor Learning Rate Sweep

Trains 3 models with different predictor learning rates to find the
setting that best encodes cart position in the predictor output.

Usage:
    python -m pendulum.train_sweep
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pendulum.model import RPLModel
from pendulum.train import EpisodeDataset, create_optimizer, train_epoch
from pendulum.evaluate import (
    collect_test_episodes,
    collect_representations,
    train_linear_decoders,
    evaluate_prediction_accuracy,
)


PREDICTOR_LRS = [3e-3, 1e-3, 3e-4]
LR_SLOW = 3e-4
EPOCHS = 50
BATCH_SIZE = 64
SEQ_LEN = 50
SEED = 42
DATA_PATH = "data/random_exploration.pkl"
NUM_TEST_EPISODES = 50
TEST_SEED = 12345


def run_single(
    predictor_lr: float,
    episodes: list,
    device: torch.device,
) -> dict:
    """Train one model and evaluate it. Returns results dict."""

    # Deterministic init
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    dataset = EpisodeDataset(episodes, seq_len=SEQ_LEN)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True,
    )

    model = RPLModel()
    model.to(device)
    optimizer = create_optimizer(model, LR_SLOW, predictor_lr)

    # Train
    losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, dataloader, optimizer, device)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.6f}")

    # Evaluate on held-out episodes
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    test_episodes = collect_test_episodes(
        num_episodes=NUM_TEST_EPISODES, seed=TEST_SEED,
    )

    # LSTM hidden state probing
    representations, next_states = collect_representations(model, test_episodes, device)
    lstm_results = train_linear_decoders(representations, next_states)

    # Predictor output probing
    pred_results = evaluate_prediction_accuracy(model, test_episodes, device)

    return {
        "predictor_lr": predictor_lr,
        "final_loss": losses[-1],
        "losses": losses,
        "lstm_r2": {name: data["r2"] for name, data in lstm_results.items()},
        "pred_r2": {name: data["r2"] for name, data in pred_results.items()},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Predictor Learning Rate Sweep ===")
    print(f"Predictor LRs: {PREDICTOR_LRS}")
    print(f"Encoder/Integrator LR: {LR_SLOW}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}, Seed: {SEED}")
    print(f"Device: {device}")

    # Load training data once
    print(f"\nLoading training data from {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes")

    # Run each configuration
    all_results = []
    for i, plr in enumerate(PREDICTOR_LRS):
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{len(PREDICTOR_LRS)}: predictor_lr = {plr}")
        print(f"{'='*60}")
        result = run_single(plr, episodes, device)
        all_results.append(result)

    # Print comparison table
    state_names = list(all_results[0]["lstm_r2"].keys())

    print(f"\n\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")

    header = f"{'Pred LR':>10} | {'Loss':>10} |"
    for name in state_names:
        short = name.split(" ")[0]
        header += f" LSTM {short:>5} |"
    for name in state_names:
        short = name.split(" ")[0]
        header += f" Pred {short:>5} |"
    print(header)
    print("-" * len(header))

    for r in all_results:
        row = f"{r['predictor_lr']:>10.0e} | {r['final_loss']:>10.6f} |"
        for name in state_names:
            row += f" {r['lstm_r2'][name]:>10.4f} |"
        for name in state_names:
            row += f" {r['pred_r2'][name]:>10.4f} |"
        print(row)

    print("-" * len(header))

    # Determine winner by predictor R2 for cart position x
    x_var = state_names[0]  # 'x (position)'
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]["pred_r2"][x_var])
    best = all_results[best_idx]

    print(f"\nBest predictor R² for {x_var}:")
    for r in all_results:
        marker = " <-- BEST" if r is best else ""
        print(f"  lr={r['predictor_lr']:.0e}: {r['pred_r2'][x_var]:.4f}{marker}")

    # Compute margin over baseline (first entry, lr=3e-3)
    baseline_r2 = all_results[0]["pred_r2"][x_var]
    best_r2 = best["pred_r2"][x_var]
    margin = best_r2 - baseline_r2
    print(f"\nWinner: predictor_lr = {best['predictor_lr']:.0e}")
    print(f"  Cart-x predictor R²: {best_r2:.4f}")
    if margin > 0:
        print(f"  Improvement over 3e-3 baseline: +{margin:.4f}")
    elif margin < 0:
        print(f"  Difference from 3e-3 baseline: {margin:.4f}")
    else:
        print(f"  Same as 3e-3 baseline")

    # Save best model
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "rpl_model_best_cart.pt"
    checkpoint = {
        "epoch": EPOCHS,
        "model_state_dict": best["model_state_dict"],
        "optimizer_state_dict": best["optimizer_state_dict"],
        "loss": best["final_loss"],
        "epoch_losses": best["losses"],
        "predictor_lr": best["predictor_lr"],
    }
    torch.save(checkpoint, save_path)
    print(f"\nBest model saved to: {save_path}")


if __name__ == "__main__":
    main()
