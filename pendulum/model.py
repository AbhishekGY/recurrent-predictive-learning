"""
RPL Network Architecture for Inverted Pendulum

This module implements the three-component architecture for Recurrent Predictive Learning:
1. Encoder: Maps raw state to embedding space
2. Integrator: LSTM that maintains temporal context (embedding-only input)
3. Predictor: Predicts next STATE embedding from current context

The network learns a forward model of the pendulum dynamics purely from
prediction errors, without explicit physics knowledge or reward signals.
Passive observation mode — no actions/forces in the model.
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


class CNNEncoder(nn.Module):
    """
    CNN encoder that maps a grayscale image to embedding space.

    Architecture:
        6 conv layers (32 filters, 5×5, padding 2) + BatchNorm + ReLU each.
        Stride 2 on layers 1-2, stride 1 on layers 3-6.
        Flatten → Linear → 32D embedding.

    Produces the same output interface as the MLP Encoder.
    """

    def __init__(self, image_size: int = 64, embedding_dim: int = 32):
        super().__init__()
        self.image_size = image_size

        layers: list[nn.Module] = []
        in_ch = 1  # grayscale
        for i in range(6):
            stride = 2 if i < 2 else 1
            layers.extend([
                nn.Conv2d(in_ch, 32, kernel_size=5, stride=stride, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ])
            in_ch = 32
        self.conv = nn.Sequential(*layers)

        # Compute flattened size by running a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            conv_out = self.conv(dummy)
            self._flat_size = conv_out.numel()

        self.fc = nn.Linear(self._flat_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image(s) to embedding(s).

        Args:
            x: Tensor of shape (batch, 1, H, W) or (batch, seq_len, 1, H, W).

        Returns:
            Embedding of shape (batch, 32) or (batch, seq_len, 32).
        """
        if x.dim() == 5:
            # Sequence input: (batch, seq_len, 1, H, W)
            b, s, c, h, w = x.shape
            x = x.reshape(b * s, c, h, w)
            out = self.conv(x)
            out = out.reshape(b * s, -1)
            out = self.fc(out)
            return out.reshape(b, s, -1)
        else:
            # Single timestep: (batch, 1, H, W)
            out = self.conv(x)
            out = out.reshape(out.size(0), -1)
            return self.fc(out)


class Integrator(nn.Module):
    """
    LSTM-based temporal integrator that maintains context over time.

    Architecture:
        Single-layer LSTM with 64 hidden units
        Input: embedding (32D)

    The integrator accumulates information over the sequence, capturing
    patterns like whether the pendulum is accelerating or decelerating.

    Mathematical operation:
        h_t, c_t = LSTM(z_t, h_{t-1}, c_{t-1})
    """

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # 32D input
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Process embedding sequence through LSTM.

        Args:
            embedding: Tensor of shape (batch, seq_len, 32) or (batch, 32)
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
            squeeze_output = True

        batch_size = embedding.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, embedding.device)

        # Run LSTM
        output, (h_n, c_n) = self.lstm(embedding, hidden)

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

    The model processes state sequences from passive observation and learns to
    predict future state embeddings. No actions/forces — purely passive dynamics.

    Training uses stop-gradient on target embeddings to prevent representation collapse.

    Network wiring at each timestep during training:
        1. Encoder processes current state -> embedding z_t = phi(x_t)
        2. Integrator accumulates temporal context -> h_t = LSTM(z_t, h_{t-1})
        3. Predictor uses representation -> predicted next embedding z_hat_{t+1} = f(h_t)
        4. Target is z_{t+1} = phi(x_{t+1}) with stop-gradient
    """

    def __init__(
        self,
        state_dim: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        use_image: bool = False,
        image_size: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_image = use_image

        if use_image:
            self.encoder = CNNEncoder(image_size, embedding_dim)
        else:
            self.encoder = Encoder(state_dim, hidden_dim, embedding_dim)
        self.integrator = Integrator(embedding_dim, hidden_dim)
        self.predictor = Predictor(hidden_dim, embedding_dim)

    def forward(
        self,
        states: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            states: Tensor of shape (batch, seq_len, 4) for MLP encoder
                    or (batch, seq_len, 1, H, W) for CNN encoder.
            hidden: Optional initial hidden state for LSTM

        Returns:
            Dictionary containing:
                - 'embeddings': Encoded states (batch, seq_len, 32)
                - 'predictions': Predicted next embeddings (batch, seq_len-1, 32)
                - 'targets': Target embeddings with stop-gradient (batch, seq_len-1, 32)
                - 'lstm_outputs': LSTM hidden states (batch, seq_len-1, 64)
                - 'hidden': Final LSTM hidden state tuple
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Encode all states
        embeddings = self.encoder(states)  # (batch, seq_len, 32)

        # For LSTM, we process states[:-1] to predict states[1:]
        lstm_outputs, hidden = self.integrator(
            embeddings[:, :-1, :],  # (batch, seq_len-1, 32)
            hidden
        )  # Output: (batch, seq_len-1, 64)

        # Predict next embeddings
        predictions = self.predictor(lstm_outputs)  # (batch, seq_len-1, 32)

        # Targets are the actual next embeddings (states[1:]) with stop-gradient
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
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step forward pass.

        Args:
            state: Tensor of shape (batch, 4) for MLP or (batch, 1, H, W) for CNN.
            hidden: LSTM hidden state from previous step

        Returns:
            Tuple of:
                - embedding: Current state embedding (batch, 32)
                - prediction: Predicted next embedding (batch, 32)
                - hidden: Updated LSTM hidden state
        """
        embedding = self.encoder(state)  # (batch, 32)
        lstm_output, hidden = self.integrator(embedding, hidden)  # (batch, 64)
        prediction = self.predictor(lstm_output)  # (batch, 32)
        return embedding, prediction, hidden

    def get_representation(
        self,
        state: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Get internal representation for a state (for linear probing).

        Args:
            state: Tensor of shape (batch, 4) for MLP or (batch, 1, H, W) for CNN.
            hidden: LSTM hidden state

        Returns:
            Tuple of:
                - representation: LSTM output (batch, 64)
                - hidden: Updated hidden state
        """
        embedding = self.encoder(state)
        lstm_output, hidden = self.integrator(embedding, hidden)
        return lstm_output, hidden


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
    print("=== RPL Model Test (Passive Observation) ===\n")

    # Create model
    model = RPLModel()
    print(f"Model created with:")
    print(f"  - State dim: {model.state_dim}")
    print(f"  - Embedding dim: {model.embedding_dim}")
    print(f"  - Hidden dim: {model.hidden_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}\n")

    # Test 1: Forward pass with sequence
    print("Test 1: Sequence forward pass")
    batch_size, seq_len = 8, 50
    dummy_states = torch.randn(batch_size, seq_len, 4)

    output = model(dummy_states)

    print(f"  States shape: {dummy_states.shape}")
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
    embedding, prediction, hidden = model.forward_step(single_state)

    print(f"  State shape: {single_state.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Hidden h shape: {hidden[0].shape}")
    print(f"  Hidden c shape: {hidden[1].shape}")
    print("  PASSED\n")

    # Test 3: Sequential single steps
    print("Test 3: Sequential observation loop")
    hidden = None
    states_sequence = torch.randn(10, 4)  # 10 timesteps

    for t in range(10):
        state = states_sequence[t:t+1]  # (1, 4)
        embedding, prediction, hidden = model.forward_step(state, hidden)

    print(f"  Processed 10 sequential steps")
    print(f"  Final embedding shape: {embedding.shape}")
    print(f"  Hidden state preserved across steps")
    print("  PASSED\n")

    # Test 4: Loss computation
    print("Test 4: Loss computation")
    output = model(dummy_states)
    loss = compute_prediction_loss(output['predictions'], output['targets'])

    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss is scalar: {loss.dim() == 0}")
    print("  PASSED\n")

    # Test 5: Gradient flow
    print("Test 5: Gradient flow check")
    model.zero_grad()
    output = model(dummy_states)
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

    print("=== All MLP tests passed ===\n")

    # --- CNN Encoder Tests ---
    print("=== CNN Encoder Tests ===\n")

    cnn_enc = CNNEncoder(image_size=64, embedding_dim=32)
    print(f"CNN flattened conv output size: {cnn_enc._flat_size}")
    cnn_params = sum(p.numel() for p in cnn_enc.parameters())
    print(f"CNN encoder parameters: {cnn_params:,}\n")

    # Test CNN-1: Single image batch
    print("CNN Test 1: Single image batch")
    dummy_img = torch.randn(4, 1, 64, 64)
    emb = cnn_enc(dummy_img)
    print(f"  Input: {dummy_img.shape} -> Output: {emb.shape}")
    assert emb.shape == (4, 32), f"Expected (4, 32), got {emb.shape}"
    print("  PASSED\n")

    # Test CNN-2: Sequence of images
    print("CNN Test 2: Image sequence")
    dummy_seq = torch.randn(4, 10, 1, 64, 64)
    emb_seq = cnn_enc(dummy_seq)
    print(f"  Input: {dummy_seq.shape} -> Output: {emb_seq.shape}")
    assert emb_seq.shape == (4, 10, 32), f"Expected (4, 10, 32), got {emb_seq.shape}"
    print("  PASSED\n")

    # Test CNN-3: Full RPLModel with use_image=True
    print("CNN Test 3: Full RPLModel forward pass")
    cnn_model = RPLModel(use_image=True)
    cnn_total = sum(p.numel() for p in cnn_model.parameters())
    print(f"  CNN RPL model parameters: {cnn_total:,}")

    img_states = torch.randn(4, 20, 1, 64, 64)
    cnn_out = cnn_model(img_states)
    print(f"  Input: {img_states.shape}")
    print(f"  Predictions: {cnn_out['predictions'].shape}")
    print(f"  Targets: {cnn_out['targets'].shape}")
    assert cnn_out['predictions'].shape == (4, 19, 32)
    print("  PASSED\n")

    # Test CNN-4: Loss and gradient flow
    print("CNN Test 4: Loss + gradient flow")
    cnn_model.zero_grad()
    cnn_out = cnn_model(img_states)
    loss = compute_prediction_loss(cnn_out['predictions'], cnn_out['targets'])
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    conv_grad = list(cnn_model.encoder.conv.parameters())[0].grad
    print(f"  CNN conv[0] gradient norm: {conv_grad.norm().item():.4f}")
    assert conv_grad is not None and conv_grad.norm().item() > 0
    print("  PASSED\n")

    # Test CNN-5: Single-step forward
    print("CNN Test 5: Single-step forward")
    single_img = torch.randn(2, 1, 64, 64)
    emb, pred, hid = cnn_model.forward_step(single_img)
    print(f"  Embedding: {emb.shape}, Prediction: {pred.shape}")
    assert emb.shape == (2, 32) and pred.shape == (2, 32)
    print("  PASSED\n")

    print("=== All CNN tests passed ===")


if __name__ == "__main__":
    test_model()
