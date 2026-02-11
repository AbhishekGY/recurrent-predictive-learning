"""
Evaluation Script for RPL Model - Linear Probing

Verifies that the trained model has learned meaningful representations
by testing whether state variables can be linearly decoded from the
internal representations.

Usage:
    python -m pendulum.evaluate --checkpoint checkpoints/rpl_model_final.pt
"""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pendulum.environment import InvertedPendulum
from pendulum.model import RPLModel


def collect_test_episodes(
    num_episodes: int = 200,
    max_steps: int = 200,
    angle_range: float = 0.2,
    omega_range: float = 0.2,
    seed: int = 12345,  # Different from training seed
) -> list:
    """
    Collect fresh test episodes for evaluation.

    Uses a different seed than training to ensure held-out data.
    """
    np.random.seed(seed)

    env = InvertedPendulum()
    episodes = []

    for i in range(num_episodes):
        env.reset(
            angle_range=angle_range,
            omega_range=omega_range,
            x_range=0.0,
            vx_range=0.0,
        )

        states = [env.get_state().copy()]
        forces = []

        done = False
        step = 0

        while not done and step < max_steps:
            force = np.random.uniform(-10, 10)
            forces.append([force])

            state, done = env.step(force)
            states.append(state.copy())
            step += 1

        episodes.append({
            'states': np.array(states, dtype=np.float32),
            'forces': np.array(forces, dtype=np.float32),
        })

    return episodes


def collect_representations(
    model: RPLModel,
    episodes: list,
    device: torch.device,
) -> tuple:
    """
    Run the trained model on episodes and collect internal representations.

    For each timestep t, we collect:
    - The LSTM hidden state h_t (after processing state_t and action_t)
    - The corresponding next state state_{t+1} (ground truth for decoding)

    Args:
        model: Trained RPL model (frozen)
        episodes: List of episode dicts
        device: Device to run on

    Returns:
        Tuple of:
            - representations: np.array of shape (N, hidden_dim)
            - next_states: np.array of shape (N, 4)
    """
    model.eval()

    all_representations = []
    all_next_states = []

    with torch.no_grad():
        for episode in episodes:
            states = episode['states']  # (T+1, 4)
            forces = episode['forces']  # (T, 1)
            T = len(forces)

            hidden = None

            for t in range(T):
                state_t = torch.tensor(states[t:t+1], dtype=torch.float32, device=device)
                action_t = torch.tensor(forces[t:t+1], dtype=torch.float32, device=device)

                # Get representation after processing (state_t, action_t)
                representation, hidden = model.get_representation(state_t, action_t, hidden)

                # Store representation and corresponding next state
                all_representations.append(representation.cpu().numpy().flatten())
                all_next_states.append(states[t + 1])

    return np.array(all_representations), np.array(all_next_states)


def train_linear_decoders(
    representations: np.ndarray,
    states: np.ndarray,
) -> dict:
    """
    Train linear regression models to decode each state variable.

    Args:
        representations: (N, hidden_dim) array of LSTM hidden states
        states: (N, 4) array of ground truth states

    Returns:
        Dictionary with decoder models and R² scores for each variable
    """
    state_names = ['x (position)', 'v_x (velocity)', 'theta (angle)', 'omega (ang. vel.)']
    results = {}

    for i, name in enumerate(state_names):
        # Train linear regression
        decoder = LinearRegression()
        decoder.fit(representations, states[:, i])

        # Predict and compute R²
        predictions = decoder.predict(representations)
        r2 = r2_score(states[:, i], predictions)

        results[name] = {
            'decoder': decoder,
            'r2': r2,
            'predictions': predictions,
            'targets': states[:, i],
        }

    return results


def evaluate_prediction_accuracy(
    model: RPLModel,
    episodes: list,
    device: torch.device,
) -> dict:
    """
    Evaluate how well the predictor's output can decode next states.

    This verifies the forward model by checking if predicted embeddings
    contain information about the actual next state.
    """
    model.eval()

    all_predictions = []
    all_next_states = []

    with torch.no_grad():
        for episode in episodes:
            states = episode['states']
            forces = episode['forces']
            T = len(forces)

            hidden = None

            for t in range(T):
                state_t = torch.tensor(states[t:t+1], dtype=torch.float32, device=device)
                action_t = torch.tensor(forces[t:t+1], dtype=torch.float32, device=device)

                # Get prediction for next embedding
                _, prediction, hidden = model.forward_step(state_t, action_t, hidden)

                all_predictions.append(prediction.cpu().numpy().flatten())
                all_next_states.append(states[t + 1])

    predictions = np.array(all_predictions)
    next_states = np.array(all_next_states)

    # Train linear decoders from predictions to next states
    return train_linear_decoders(predictions, next_states)


def print_results(results: dict, title: str) -> None:
    """Print R² results in a formatted table."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"{'State Variable':<25} {'R² Score':<10} {'Status'}")
    print("-" * 50)

    all_pass = True
    for name, data in results.items():
        r2 = data['r2']
        if r2 >= 0.9:
            status = "✓ Excellent"
        elif r2 >= 0.7:
            status = "✓ Good"
        elif r2 >= 0.5:
            status = "~ Marginal"
            all_pass = False
        else:
            status = "✗ Poor"
            all_pass = False

        print(f"{name:<25} {r2:>8.4f}   {status}")

    print("-" * 50)
    mean_r2 = np.mean([d['r2'] for d in results.values()])
    print(f"{'Mean R²':<25} {mean_r2:>8.4f}")

    if all_pass:
        print("\n✓ All state variables are linearly decodable (R² >= 0.7)")
    else:
        print("\n✗ Some state variables have low decodability")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RPL model via linear probing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rpl_model_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of test episodes to collect (default: 200)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for test data (default: 12345, different from training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== RPL Model Evaluation (Linear Probing) ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test episodes: {args.num_episodes}")
    print(f"Test seed: {args.seed}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = RPLModel()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Freeze parameters (not strictly necessary in eval mode, but explicit)
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss was: {checkpoint.get('loss', 'unknown'):.6f}")

    # Collect test episodes
    print(f"\nCollecting {args.num_episodes} test episodes...")
    episodes = collect_test_episodes(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    total_transitions = sum(len(ep['forces']) for ep in episodes)
    print(f"Collected {total_transitions:,} transitions")

    # Collect representations
    print("\nCollecting internal representations...")
    representations, next_states = collect_representations(model, episodes, device)
    print(f"Representation shape: {representations.shape}")

    # Train linear decoders and evaluate
    print("\nTraining linear decoders...")
    repr_results = train_linear_decoders(representations, next_states)
    print_results(repr_results, "Linear Probing: LSTM Hidden State -> Next State")

    # Also evaluate predictor output
    print("\nEvaluating predictor output...")
    pred_results = evaluate_prediction_accuracy(model, episodes, device)
    print_results(pred_results, "Linear Probing: Predicted Embedding -> Next State")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    repr_mean = np.mean([d['r2'] for d in repr_results.values()])
    pred_mean = np.mean([d['r2'] for d in pred_results.values()])

    print(f"Mean R² from LSTM hidden state: {repr_mean:.4f}")
    print(f"Mean R² from predicted embedding: {pred_mean:.4f}")

    if repr_mean >= 0.7 and pred_mean >= 0.7:
        print("\n✓ Model has learned meaningful representations of pendulum dynamics!")
    elif repr_mean >= 0.7:
        print("\n~ LSTM captures dynamics but predictor may need more training")
    else:
        print("\n✗ Model may need more training or architectural changes")


if __name__ == "__main__":
    main()
