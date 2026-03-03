"""
Pendulum Renderer — renders the cart-pendulum system as a 64×64 grayscale image.

Produces lightweight numpy-only rasterized frames suitable for CNN input.
No matplotlib dependency — uses direct pixel operations for speed.

Usage:
    python -m pendulum.render          # save test frames to plots/render_test/
"""

import numpy as np


# Physical constants (must match environment.py)
_PENDULUM_LENGTH: float = 0.5   # meters
_CART_WIDTH: float = 0.5        # meters
_CART_HEIGHT: float = 0.15      # meters
_BOB_RADIUS: float = 0.06       # meters

# Viewport: square, uniform scaling, centered on cart each frame
_VIEW_HALF: float = 1.0         # half-size in meters (2m × 2m window)
_X_MIN: float = -_VIEW_HALF
_X_MAX: float = _VIEW_HALF
_Y_MIN: float = -_VIEW_HALF
_Y_MAX: float = _VIEW_HALF

# Pixel intensities
_BG_VALUE: float = 0.2
_FG_VALUE: float = 1.0


def _world_to_pixel(
    wx: float, wy: float, image_size: int
) -> tuple[float, float]:
    """Convert world coordinates to pixel coordinates.

    Args:
        wx: World x coordinate (meters).
        wy: World y coordinate (meters).
        image_size: Image dimension in pixels.

    Returns:
        (px, py) pixel coordinates (float, not clipped).
    """
    px = (wx - _X_MIN) / (_X_MAX - _X_MIN) * image_size
    # Flip y — world y-up maps to pixel row-down
    py = (1.0 - (wy - _Y_MIN) / (_Y_MAX - _Y_MIN)) * image_size
    return px, py


def _draw_filled_rect(
    img: np.ndarray,
    cx: float, cy: float,
    half_w: float, half_h: float,
    value: float,
    image_size: int,
) -> None:
    """Draw an axis-aligned filled rectangle in pixel space (in-place)."""
    px, py = _world_to_pixel(cx, cy, image_size)
    hw_px = half_w / (_X_MAX - _X_MIN) * image_size
    hh_px = half_h / (_Y_MAX - _Y_MIN) * image_size

    r_min = max(0, int(py - hh_px))
    r_max = min(image_size, int(py + hh_px) + 1)
    c_min = max(0, int(px - hw_px))
    c_max = min(image_size, int(px + hw_px) + 1)

    img[r_min:r_max, c_min:c_max] = value


def _draw_filled_circle(
    img: np.ndarray,
    wx: float, wy: float,
    radius_world: float,
    value: float,
    image_size: int,
) -> None:
    """Draw a filled circle at world coordinates (in-place)."""
    px, py = _world_to_pixel(wx, wy, image_size)
    r_px = radius_world / (_X_MAX - _X_MIN) * image_size

    # Bounding box
    r_min = max(0, int(py - r_px) - 1)
    r_max = min(image_size, int(py + r_px) + 2)
    c_min = max(0, int(px - r_px) - 1)
    c_max = min(image_size, int(px + r_px) + 2)

    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            dist_sq = (c + 0.5 - px) ** 2 + (r + 0.5 - py) ** 2
            if dist_sq <= r_px ** 2:
                img[r, c] = value


def _draw_line(
    img: np.ndarray,
    wx0: float, wy0: float,
    wx1: float, wy1: float,
    thickness_world: float,
    value: float,
    image_size: int,
) -> None:
    """Draw a thick line between two world-coordinate points (in-place).

    Uses perpendicular-distance thresholding for uniform thickness.
    """
    px0, py0 = _world_to_pixel(wx0, wy0, image_size)
    px1, py1 = _world_to_pixel(wx1, wy1, image_size)

    # Half-thickness in pixels (average x/y scale)
    scale = image_size / (_X_MAX - _X_MIN)
    ht_px = thickness_world * scale * 0.5

    # Bounding box with margin
    margin = int(ht_px) + 2
    r_min = max(0, int(min(py0, py1)) - margin)
    r_max = min(image_size, int(max(py0, py1)) + margin + 1)
    c_min = max(0, int(min(px0, px1)) - margin)
    c_max = min(image_size, int(max(px0, px1)) + margin + 1)

    dx = px1 - px0
    dy = py1 - py0
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-8:
        return

    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            # Project pixel center onto line segment
            pc = c + 0.5 - px0
            pr = r + 0.5 - py0
            t = (pc * dx + pr * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            # Perpendicular distance
            proj_c = px0 + t * dx
            proj_r = py0 + t * dy
            dist_sq = (c + 0.5 - proj_c) ** 2 + (r + 0.5 - proj_r) ** 2
            if dist_sq <= ht_px ** 2:
                img[r, c] = value


def render_pendulum(state: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Render the cart-pendulum system as a grayscale image.

    Args:
        state: Array-like of shape (4,) — [x, v_x, theta, omega].
            Only x and theta are used for rendering.
        image_size: Output image dimension (square).

    Returns:
        Image tensor of shape (1, image_size, image_size), float32, values in [0, 1].
    """
    theta = float(state[2])

    img = np.full((image_size, image_size), _BG_VALUE, dtype=np.float32)

    # Cart — always at origin (viewport follows the cart)
    _draw_filled_rect(
        img, 0.0, 0.0,
        _CART_WIDTH / 2, _CART_HEIGHT / 2,
        _FG_VALUE, image_size,
    )

    # Pole — from cart center to tip (cart-relative coordinates)
    tip_x = _PENDULUM_LENGTH * np.sin(theta)
    tip_y = _PENDULUM_LENGTH * np.cos(theta)
    _draw_line(
        img, 0.0, 0.0, tip_x, tip_y,
        thickness_world=0.06,
        value=_FG_VALUE,
        image_size=image_size,
    )

    # Bob — circle at pole tip
    _draw_filled_circle(
        img, tip_x, tip_y,
        _BOB_RADIUS, _FG_VALUE, image_size,
    )

    return img[np.newaxis, :, :]  # (1, H, W)


def test_renderer() -> None:
    """Save test frames at known states and print diagnostics."""
    from pathlib import Path

    out_dir = Path("plots/render_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        ("upright",     [0.0, 0.0, 0.0, 0.0]),
        ("right_45",    [0.0, 0.0, np.pi / 4, 0.0]),
        ("horizontal",  [0.0, 0.0, np.pi / 2, 0.0]),
        ("hanging",     [0.0, 0.0, np.pi, 0.0]),
        ("cart_right",  [1.5, 0.0, 0.3, 0.0]),
    ]

    for name, state in test_cases:
        state_arr = np.array(state, dtype=np.float32)
        img = render_pendulum(state_arr)

        assert img.shape == (1, 64, 64), f"Bad shape: {img.shape}"
        assert img.dtype == np.float32, f"Bad dtype: {img.dtype}"
        assert 0.0 <= img.min() and img.max() <= 1.0, f"Out of range: [{img.min()}, {img.max()}]"

        # Save as PNG via PIL if available, else raw numpy
        try:
            from PIL import Image
            pil_img = Image.fromarray((img[0] * 255).astype(np.uint8), mode="L")
            path = out_dir / f"{name}.png"
            pil_img.save(path)
            print(f"  Saved {path}  (min={img.min():.2f}, max={img.max():.2f})")
        except ImportError:
            path = out_dir / f"{name}.npy"
            np.save(path, img)
            print(f"  Saved {path}  (min={img.min():.2f}, max={img.max():.2f})")

    print(f"\nAll {len(test_cases)} test frames rendered successfully.")


if __name__ == "__main__":
    test_renderer()
