"""
Sanity Check for CNN-LSTM RPL Pipeline

Runs 6 checks to verify the full image-based training pipeline works end-to-end.

Usage:
    python -m pendulum.sanity_check
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pendulum.model import RPLModel, CNNEncoder, compute_prediction_loss
from pendulum.render import render_pendulum
from pendulum.train import ImageEpisodeDataset, create_optimizer


def check_render():
    """Check 1: Render 5 frames at known states and save to plots/sanity/."""
    out_dir = Path("plots/sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_states = [
        ("upright",    [0.0, 0.0, 0.0, 0.0]),
        ("tilted_R",   [0.0, 0.0, 0.5, 0.0]),
        ("tilted_L",   [0.0, 0.0, -0.5, 0.0]),
        ("cart_right", [1.0, 0.0, 0.3, 0.0]),
        ("cart_left",  [-1.0, 0.0, -0.3, 0.0]),
    ]

    for name, state in test_states:
        img = render_pendulum(np.array(state, dtype=np.float32))
        assert img.shape == (1, 64, 64), f"Bad shape: {img.shape}"
        assert img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0

        try:
            from PIL import Image
            pil_img = Image.fromarray((img[0] * 255).astype(np.uint8), mode="L")
            pil_img.save(out_dir / f"{name}.png")
        except ImportError:
            np.save(out_dir / f"{name}.npy", img)

    print(f"  Saved 5 frames to {out_dir}/")
    return True


def check_cnn_encoder_shape():
    """Check 2: CNN encoder produces (batch, 32) from image batch."""
    encoder = CNNEncoder(image_size=64, embedding_dim=32)
    x = torch.randn(4, 1, 64, 64)
    out = encoder(x)
    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"
    print(f"  Input {tuple(x.shape)} -> Output {tuple(out.shape)}")
    return True


def check_full_forward():
    """Check 3: Full RPL forward pass (CNN) produces valid scalar loss."""
    model = RPLModel(use_image=True)
    images = torch.randn(2, 11, 1, 64, 64)  # batch=2, seq_len=11
    output = model(images)
    loss = compute_prediction_loss(output['predictions'], output['targets'])
    assert loss.dim() == 0, f"Loss not scalar: dim={loss.dim()}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    print(f"  Predictions: {tuple(output['predictions'].shape)}, Loss: {loss.item():.4f}")
    return True


def check_backward_gradients():
    """Check 4: Gradients flow to CNN encoder weights after backward pass."""
    model = RPLModel(use_image=True)
    images = torch.randn(2, 11, 1, 64, 64)

    model.zero_grad()
    output = model(images)
    loss = compute_prediction_loss(output['predictions'], output['targets'])
    loss.backward()

    conv_param = list(model.encoder.conv.parameters())[0]
    assert conv_param.grad is not None, "No gradient on conv layer"
    grad_norm = conv_param.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is zero"
    print(f"  CNN conv[0] grad norm: {grad_norm:.6f}")
    return True


def check_training_loop():
    """Check 5: Loss decreases over 100 steps on small generated dataset."""
    # Generate a small synthetic dataset
    episodes = []
    for _ in range(32):
        T = 20
        states = np.random.randn(T + 1, 4).astype(np.float32) * 0.5
        states[:, 0] = np.clip(states[:, 0], -1.0, 1.0)  # keep x in range
        states[:, 2] = np.clip(states[:, 2], -1.0, 1.0)  # keep theta in range
        images = np.stack([render_pendulum(s) for s in states])  # (T+1, 1, 64, 64)
        episodes.append({'images': images, 'states': states})

    dataset = ImageEpisodeDataset(episodes, seq_len=15)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    model = RPLModel(use_image=True)
    optimizer = create_optimizer(model, lr_slow=3e-4, lr_fast=3e-3)

    losses = []
    model.train()
    for step in range(100):
        for batch_images, mask in dataloader:
            output = model(batch_images)
            loss = compute_prediction_loss(output['predictions'], output['targets'], mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

    first_10 = np.mean(losses[:10])
    last_10 = np.mean(losses[-10:])
    decreased = last_10 < first_10
    print(f"  First 10 avg: {first_10:.4f}, Last 10 avg: {last_10:.4f}, Decreased: {decreased}")
    return decreased


def check_print_summary(results):
    """Check 6: Print pass/fail summary for all checks."""
    print("\n" + "=" * 50)
    print("SANITY CHECK SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
    print("=" * 50)
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")
    return all_pass


def main():
    checks = [
        ("Render 5 frames", check_render),
        ("CNN encoder output shape", check_cnn_encoder_shape),
        ("Full forward pass + loss", check_full_forward),
        ("Backward pass gradient flow", check_backward_gradients),
        ("100-step training loss decrease", check_training_loop),
    ]

    results = []
    for name, fn in checks:
        print(f"\nCheck: {name}")
        try:
            passed = fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    all_pass = check_print_summary(results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
