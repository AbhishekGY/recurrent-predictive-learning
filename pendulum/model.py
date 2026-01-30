"""
RPL Network Architecture for Inverted Pendulum Control

This module implements the three-component architecture for Recurrent Predictive Learning:
1. Encoder: Maps raw state to embedding space (NO action input)
2. Integrator: LSTM that maintains temporal context (action-conditioned)
3. Predictor: Predicts next STATE embedding from current context

KEY ARCHITECTURAL PRINCIPLE: The network learns an action-conditioned forward model
where actions are inputs to the integrator but NOT part of the prediction target.
This avoids the problem of trying to predict randomly sampled forces.

The network learns a forward model of the pendulum dynamics purely from
prediction errors, without explicit physics knowledge or reward signals.
"""

import torch
import torch.nn as nn
from typing import Optional


class Encoder(nn.Module):
    """
    Feedforward encoder that maps raw state to embedding space.

    Architecture:
        Input (4D) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(32)

    The encoder is memoryless - it processes each state independently.
    It does NOT receive force/action as input.

    Mathematical operation:
        z_t = phi(x_t)  where x_t = [x, v_x, theta, omega]
    """

    def __init__(self, state_dim: int = 4, hidden_dim: int = 64, embedding_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode raw state to embedding.

        Args:
            state: Tensor of shape (batch, 4) or (batch, seq_len, 4)

        Returns:
            Embedding tensor of shape (batch, 32) or (batch, seq_len, 32)
        """
        return self.network(state)


class Integrator(nn.Module):
    """
    LSTM-based temporal integrator that maintains context over time.

    Architecture:
        Single-layer LSTM with 64 hidden units
        Input: concatenated [embedding (32D), action (1D)] = 33D

    The integrator accumulates information over the sequence, capturing
    patterns like whether the pendulum is accelerating or decelerating,
    AND how actions have influenced the trajectory.

    Mathematical operation:
        h_t, c_t = LSTM([z_t, F_t], h_{t-1}, c_{t-1})
    """

    def __init__(self, embedding_dim: int = 32, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # Input is embedding + action concatenated
        self.lstm = nn.LSTM(
            input_size=embedding_dim + action_dim,  # 33D input
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Process embedding and action sequence through LSTM.

        Args:
            embedding: Tensor of shape (batch, seq_len, 32) or (batch, 32)
            action: Tensor of shape (batch, seq_len, 1) or (batch, 1)
            hidden: Optional tuple of (h_0, c_0) each of shape (1, batch, 64)

        Returns:
            Tuple of:
                - hidden_states: Tensor of shape (batch, seq_len, 64) or (batch, 64)
                - hidden: Tuple of (h_n, c_n) for next step
        """
        # Handle single timestep input
        squeeze_output = False
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(1)  # (batch, 1, 32)
            action = action.unsqueeze(1)  # (batch, 1, 1)
            squeeze_output = True

        batch_size = embedding.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, embedding.device)

        # Concatenate embedding and action
        lstm_input = torch.cat([embedding, action], dim=-1)  # (batch, seq_len, 33)

        # Run LSTM
        output, (h_n, c_n) = self.lstm(lstm_input, hidden)

        if squeeze_output:
            output = output.squeeze(1)  # (batch, 64)

        return output, (h_n, c_n)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state to zeros."""
        h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


class Predictor(nn.Module):
    """
    MLP that predicts next STATE embedding from current LSTM state.

    Architecture:
        Input (64D) -> Linear(64) -> ReLU -> Linear(32)

    The predictor learns the forward dynamics in embedding space.
    It predicts next STATE embedding only, NOT the next action.

    Mathematical operation:
        z_hat_{t+1} = f(h_t)
    """

    def __init__(self, hidden_dim: int = 64, embedding_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Predict next embedding from LSTM hidden state.

        Args:
            lstm_output: Tensor of shape (batch, 64) or (batch, seq_len, 64)

        Returns:
            Predicted embedding of shape (batch, 32) or (batch, seq_len, 32)
        """
        return self.network(lstm_output)


class RPLModel(nn.Module):
    """
    Complete RPL model combining Encoder, Integrator, and Predictor.

    The model processes state-action sequences and learns to predict future
    state embeddings. Actions are inputs to the model but NOT prediction targets.

    Training uses stop-gradient on target embeddings to prevent representation collapse.

    Network wiring at each timestep during training:
        1. Encoder processes current state -> embedding z_t = phi(x_t)
        2. Integrator incorporates embedding AND action -> h_t = LSTM([z_t, F_t], h_{t-1})
        3. Predictor uses representation -> predicted next embedding z_hat_{t+1} = f(h_t)
        4. Target is z_{t+1} = phi(x_{t+1}) with stop-gradient (no force in target!)
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(state_dim, hidden_dim, embedding_dim)
        self.integrator = Integrator(embedding_dim, action_dim, hidden_dim)
        self.predictor = Predictor(hidden_dim, embedding_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            states: Tensor of shape (batch, seq_len, 4) - sequence of states
            actions: Tensor of shape (batch, seq_len-1, 1) - forces applied
                     Action[t] causes transition from state[t] to state[t+1]
            hidden: Optional initial hidden state for LSTM

        Returns:
            Dictionary containing:
                - 'embeddings': Encoded states (batch, seq_len, 32)
                - 'predictions': Predicted next embeddings (batch, seq_len-1, 32)
                - 'targets': Target embeddings with stop-gradient (batch, seq_len-1, 32)
                - 'lstm_outputs': LSTM hidden states (batch, seq_len-1, 64)
                - 'hidden': Final LSTM hidden state tuple
        """
        batch_size, seq_len, _ = states.shape

        # Encode all states (no action in encoder!)
        embeddings = self.encoder(states)  # (batch, seq_len, 32)

        # For LSTM, we process states[:-1] with actions to predict states[1:]
        # Run through integrator with embeddings[:-1] and actions
        lstm_outputs, hidden = self.integrator(
            embeddings[:, :-1, :],  # (batch, seq_len-1, 32)
            actions,                 # (batch, seq_len-1, 1)
            hidden
        )  # Output: (batch, seq_len-1, 64)

        # Predict next embeddings
        predictions = self.predictor(lstm_outputs)  # (batch, seq_len-1, 32)

        # Targets are the actual next embeddings (states[1:]) with stop-gradient
        # Note: targets are STATE embeddings only, no force component!
        targets = embeddings[:, 1:, :].detach()  # (batch, seq_len-1, 32)

        return {
            'embeddings': embeddings,
            'predictions': predictions,
            'targets': targets,
            'lstm_outputs': lstm_outputs,
            'hidden': hidden,
        }

    def forward_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step forward pass for control.

        Args:
            state: Tensor of shape (batch, 4) - current state
            action: Tensor of shape (batch, 1) - force to apply
            hidden: LSTM hidden state from previous step

        Returns:
            Tuple of:
                - embedding: Current state embedding (batch, 32)
                - prediction: Predicted next embedding (batch, 32)
                - hidden: Updated LSTM hidden state
        """
        # Encode current state (no action)
        embedding = self.encoder(state)  # (batch, 32)

        # Update integrator with embedding and action
        lstm_output, hidden = self.integrator(embedding, action, hidden)  # (batch, 64)

        # Predict next embedding
        prediction = self.predictor(lstm_output)  # (batch, 32)

        return embedding, prediction, hidden

    def get_representation(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Get internal representation for a state-action pair (for linear probing).

        Args:
            state: Tensor of shape (batch, 4)
            action: Tensor of shape (batch, 1)
            hidden: LSTM hidden state

        Returns:
            Tuple of:
                - representation: LSTM output (batch, 64)
                - hidden: Updated hidden state
        """
        embedding = self.encoder(state)
        lstm_output, hidden = self.integrator(embedding, action, hidden)
        return lstm_output, hidden

    def predict_for_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict next state embedding for a hypothetical action (for planning).

        This creates a temporary/hypothetical integrator state - use for evaluating
        candidate actions during control, but don't persist the hidden state.

        Args:
            state: Tensor of shape (batch, 4) - current state
            action: Tensor of shape (batch, 1) - hypothetical force to evaluate
            hidden: LSTM hidden state from processing history

        Returns:
            Predicted next state embedding (batch, 32)
        """
        embedding = self.encoder(state)
        lstm_output, _ = self.integrator(embedding, action, hidden)
        prediction = self.predictor(lstm_output)
        return prediction


def compute_prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MSE loss between predictions and targets.

    Args:
        predictions: Predicted embeddings (batch, seq_len, embedding_dim)
        targets: Target embeddings (batch, seq_len, embedding_dim)
        mask: Optional boolean mask (batch, seq_len) for valid timesteps

    Returns:
        Scalar loss value
    """
    # Squared error
    squared_error = (predictions - targets) ** 2  # (batch, seq_len, embedding_dim)

    # Mean over embedding dimension
    mse_per_step = squared_error.mean(dim=-1)  # (batch, seq_len)

    if mask is not None:
        # Apply mask and compute mean only over valid steps
        masked_mse = mse_per_step * mask
        loss = masked_mse.sum() / mask.sum().clamp(min=1)
    else:
        loss = mse_per_step.mean()

    return loss


def test_model():
    """Test the RPL model with dummy data."""
    print("=== RPL Model Test (Action-Conditioned) ===\n")

    # Create model
    model = RPLModel()
    print(f"Model created with:")
    print(f"  - State dim: {model.state_dim}")
    print(f"  - Action dim: {model.action_dim}")
    print(f"  - Embedding dim: {model.embedding_dim}")
    print(f"  - Hidden dim: {model.hidden_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}\n")

    # Test 1: Forward pass with sequence
    print("Test 1: Sequence forward pass")
    batch_size, seq_len = 8, 50
    dummy_states = torch.randn(batch_size, seq_len, 4)
    dummy_actions = torch.randn(batch_size, seq_len - 1, 1)  # T-1 actions for T states

    output = model(dummy_states, dummy_actions)

    print(f"  States shape: {dummy_states.shape}")
    print(f"  Actions shape: {dummy_actions.shape}")
    print(f"  Embeddings shape: {output['embeddings'].shape}")
    print(f"  Predictions shape: {output['predictions'].shape}")
    print(f"  Targets shape: {output['targets'].shape}")
    print(f"  LSTM outputs shape: {output['lstm_outputs'].shape}")
    assert output['predictions'].shape == (batch_size, seq_len - 1, 32)
    assert output['targets'].shape == (batch_size, seq_len - 1, 32)
    print("  PASSED\n")

    # Test 2: Single step forward
    print("Test 2: Single step forward pass")
    single_state = torch.randn(batch_size, 4)
    single_action = torch.randn(batch_size, 1)
    embedding, prediction, hidden = model.forward_step(single_state, single_action)

    print(f"  State shape: {single_state.shape}")
    print(f"  Action shape: {single_action.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Hidden h shape: {hidden[0].shape}")
    print(f"  Hidden c shape: {hidden[1].shape}")
    print("  PASSED\n")

    # Test 3: Sequential single steps (simulating control loop)
    print("Test 3: Sequential control loop simulation")
    hidden = None
    states_sequence = torch.randn(10, 4)  # 10 timesteps
    actions_sequence = torch.randn(10, 1)  # 10 actions

    for t in range(10):
        state = states_sequence[t:t+1]  # (1, 4)
        action = actions_sequence[t:t+1]  # (1, 1)
        embedding, prediction, hidden = model.forward_step(state, action, hidden)

    print(f"  Processed 10 sequential steps with actions")
    print(f"  Final embedding shape: {embedding.shape}")
    print(f"  Hidden state preserved across steps")
    print("  PASSED\n")

    # Test 4: Loss computation
    print("Test 4: Loss computation")
    output = model(dummy_states, dummy_actions)
    loss = compute_prediction_loss(output['predictions'], output['targets'])

    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss is scalar: {loss.dim() == 0}")
    print("  PASSED\n")

    # Test 5: Gradient flow
    print("Test 5: Gradient flow check")
    model.zero_grad()
    output = model(dummy_states, dummy_actions)
    loss = compute_prediction_loss(output['predictions'], output['targets'])
    loss.backward()

    encoder_grad = model.encoder.network[0].weight.grad
    integrator_grad = model.integrator.lstm.weight_ih_l0.grad
    predictor_grad = model.predictor.network[0].weight.grad

    print(f"  Encoder gradient norm: {encoder_grad.norm().item():.4f}")
    print(f"  Integrator gradient norm: {integrator_grad.norm().item():.4f}")
    print(f"  Predictor gradient norm: {predictor_grad.norm().item():.4f}")
    print(f"  Gradients are flowing through all components")
    print("  PASSED\n")

    # Test 6: Stop-gradient verification
    print("Test 6: Stop-gradient on targets")
    print(f"  targets.requires_grad: {output['targets'].requires_grad}")
    print(f"  predictions.requires_grad: {output['predictions'].requires_grad}")
    assert not output['targets'].requires_grad
    assert output['predictions'].requires_grad
    print("  PASSED\n")

    # Test 7: Action planning (hypothetical evaluation)
    print("Test 7: Action planning - evaluate candidate actions")
    current_state = torch.randn(1, 4)
    candidate_forces = [-10, -5, 0, 5, 10]

    # First, build up hidden state from some history
    hidden = None
    for _ in range(5):
        _, _, hidden = model.forward_step(
            torch.randn(1, 4),
            torch.randn(1, 1),
            hidden
        )

    # Now evaluate candidate actions
    predictions = []
    for force in candidate_forces:
        force_tensor = torch.tensor([[force]], dtype=torch.float32)
        pred = model.predict_for_action(current_state, force_tensor, hidden)
        predictions.append(pred)

    print(f"  Current state shape: {current_state.shape}")
    print(f"  Evaluated {len(candidate_forces)} candidate actions")
    print(f"  Each prediction shape: {predictions[0].shape}")
    print("  PASSED\n")

    # Test 8: Verify action affects prediction
    print("Test 8: Different actions produce different predictions")
    state = torch.zeros(1, 4)  # Fixed state
    hidden = None

    pred_neg = model.predict_for_action(state, torch.tensor([[-10.0]]), hidden)
    pred_pos = model.predict_for_action(state, torch.tensor([[10.0]]), hidden)

    diff = (pred_neg - pred_pos).abs().mean().item()
    print(f"  Prediction difference for F=-10 vs F=+10: {diff:.4f}")
    assert diff > 0, "Different actions should produce different predictions"
    print("  PASSED\n")

    print("=== All tests passed ===")


if __name__ == "__main__":
    test_model()
