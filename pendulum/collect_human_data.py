"""
Human-in-the-Loop Data Collection for RPL Inverted Pendulum

The human controls the pendulum via keyboard to generate high-quality training
episodes. Data is stored in the same format as random exploration so both
datasets can be merged directly.

Controls:
    LEFT ARROW  -> apply force -F N
    RIGHT ARROW -> apply force +F N
    DOWN ARROW  -> apply force 0 N (brake)
    R           -> reset episode (saves if long enough)
    S           -> save dataset and quit
    Q           -> quit (keeps previously completed episodes)

Usage:
    python -m pendulum.collect_human_data --output_path data/human_collected.pkl
"""

import argparse
import math
import os
import pickle
import tempfile
import time

import numpy as np
import pygame

from pendulum.environment import InvertedPendulum


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 720

# Colours
BG_COLOR = (30, 30, 38)
PANEL_BG = (42, 42, 54)
TRACK_COLOR = (180, 180, 180)
CART_COLOR = (33, 150, 243)
BOB_GREEN = (76, 175, 80)
BOB_YELLOW = (255, 193, 7)
BOB_RED = (244, 67, 54)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT = (140, 140, 160)
HINT_BG = (24, 24, 30)
FLASH_SAVE = (76, 175, 80)
FLASH_DISCARD = (244, 67, 54)
BOUNDARY_COLOR = (255, 80, 80)
FORCE_BAR_POS = (60, 130, 220)
FORCE_BAR_NEG = (220, 100, 60)

# Layout
CANVAS_TOP = 140
CANVAS_HEIGHT = 360
CANVAS_CENTER_Y = CANVAS_TOP + CANVAS_HEIGHT // 2 + 40
STATS_PANEL_X = 700
HINTS_HEIGHT = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rod_color(theta: float) -> tuple:
    """Return rod colour based on angle magnitude."""
    a = abs(theta)
    if a < 0.2:
        return BOB_GREEN
    elif a < 0.5:
        return BOB_YELLOW
    return BOB_RED


def world_to_screen(x: float, track_limit: float) -> int:
    """Map world x in [-track_limit, track_limit] to screen pixel x."""
    margin = 60
    usable = SCREEN_WIDTH - STATS_PANEL_X + 300 - 2 * margin
    # Centre the canvas in the left 700px
    canvas_width = min(usable, 620)
    canvas_left = (STATS_PANEL_X - canvas_width) // 2
    frac = (x + track_limit) / (2 * track_limit)
    return int(canvas_left + frac * canvas_width)


def save_atomic(episodes: list, path: str) -> None:
    """Write episodes to *path* atomically via temp-file + rename."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(episodes, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-loop data collection for inverted pendulum"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/human_collected.pkl",
        help="Path to save collected episodes (default: data/human_collected.pkl)",
    )
    parser.add_argument(
        "--max_steps_per_episode", type=int, default=500,
        help="Max timesteps per episode (default: 500)",
    )
    parser.add_argument(
        "--min_steps_to_save", type=int, default=30,
        help="Minimum episode length to keep (default: 30)",
    )
    parser.add_argument(
        "--force_magnitude", type=float, default=10.0,
        help="Force magnitude for left/right keys (default: 10.0)",
    )
    parser.add_argument(
        "--fps", type=int, default=50,
        help="Simulation and render rate in Hz (default: 50)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resume: load existing data
    # ------------------------------------------------------------------
    all_episodes: list[dict] = []
    if os.path.exists(args.output_path):
        with open(args.output_path, "rb") as f:
            all_episodes = pickle.load(f)
        total_steps = sum(len(ep["forces"]) for ep in all_episodes)
        print(f"Resumed: {len(all_episodes)} episodes, {total_steps} timesteps "
              f"from {args.output_path}")
    else:
        print(f"Starting fresh (no file at {args.output_path})")

    episodes_at_start = len(all_episodes)

    # ------------------------------------------------------------------
    # Pygame init
    # ------------------------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RPL Human Data Collection")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("monospace", 22, bold=True)
    font_med = pygame.font.SysFont("monospace", 16)
    font_small = pygame.font.SysFont("monospace", 13)

    env = InvertedPendulum(dt=1.0 / args.fps)
    track_limit = env.track_limit

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------
    state = env.reset(angle_range=0.1, omega_range=0.1)
    ep_states = [state.copy()]
    ep_forces: list[list[float]] = []
    step = 0
    done = False
    session_completed = 0
    session_start = time.time()
    flash_msg = ""
    flash_color = FLASH_SAVE
    flash_until = 0.0
    running = True

    def finish_episode(reason: str = ""):
        """Finalise current episode, save if long enough, auto-reset."""
        nonlocal all_episodes, session_completed, flash_msg, flash_color, flash_until
        nonlocal state, ep_states, ep_forces, step, done

        if step >= args.min_steps_to_save:
            episode = {
                "states": np.array(ep_states, dtype=np.float32),
                "forces": np.array(ep_forces, dtype=np.float32),
            }
            all_episodes.append(episode)
            session_completed += 1
            flash_msg = f"Episode saved ({step} steps) {reason}"
            flash_color = FLASH_SAVE

            # Auto-save every 10 completed episodes
            if session_completed % 10 == 0:
                save_atomic(all_episodes, args.output_path)
                flash_msg += " [auto-saved]"
        else:
            flash_msg = f"Episode discarded ({step} < {args.min_steps_to_save} steps)"
            flash_color = FLASH_DISCARD

        flash_until = time.time() + 2.0

        # Reset
        state = env.reset(angle_range=0.1, omega_range=0.1)
        ep_states = [state.copy()]
        ep_forces = []
        step = 0
        done = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while running:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    finish_episode("(manual save+quit)")
                    save_atomic(all_episodes, args.output_path)
                    flash_msg = f"Dataset saved ({len(all_episodes)} episodes)"
                    flash_color = FLASH_SAVE
                    flash_until = time.time() + 2.0
                    running = False
                elif event.key == pygame.K_r:
                    finish_episode("(manual reset)")

        if not running:
            break

        # --- Determine force from held keys ---
        keys = pygame.key.get_pressed()
        force = 0.0
        if keys[pygame.K_LEFT]:
            force -= args.force_magnitude
        if keys[pygame.K_RIGHT]:
            force += args.force_magnitude
        if keys[pygame.K_DOWN]:
            force = 0.0

        # --- Physics step ---
        if not done and step < args.max_steps_per_episode:
            ep_forces.append([force])
            state, done = env.step(force)
            ep_states.append(state.copy())
            step += 1

            if done or step >= args.max_steps_per_episode:
                reason = ""
                if done:
                    x, _, theta, _ = state
                    if abs(theta) > math.pi / 2:
                        reason = "[angle]"
                    elif abs(x) >= track_limit:
                        reason = "[cart boundary]"
                finish_episode(reason)

        # ------------------------------------------------------------------
        # Render
        # ------------------------------------------------------------------
        screen.fill(BG_COLOR)

        x_pos, v_x, theta, omega = state

        # ---- Pendulum canvas ----
        # Track line
        track_y = CANVAS_CENTER_Y
        track_left = world_to_screen(-track_limit, track_limit)
        track_right = world_to_screen(track_limit, track_limit)
        pygame.draw.line(screen, TRACK_COLOR,
                         (track_left, track_y), (track_right, track_y), 3)

        # Track boundaries
        for bx in (-track_limit, track_limit):
            sx = world_to_screen(bx, track_limit)
            pygame.draw.line(screen, BOUNDARY_COLOR,
                             (sx, track_y - 50), (sx, track_y + 10), 2)

        # Cart
        cart_sx = world_to_screen(x_pos, track_limit)
        cart_w, cart_h = 50, 26
        cart_rect = pygame.Rect(cart_sx - cart_w // 2, track_y - cart_h // 2,
                                cart_w, cart_h)
        pygame.draw.rect(screen, CART_COLOR, cart_rect, border_radius=4)

        # Pendulum rod + bob
        pend_pixel_len = 140  # visual length in pixels
        bob_x = cart_sx + int(pend_pixel_len * math.sin(theta))
        bob_y = track_y - int(pend_pixel_len * math.cos(theta))

        color = rod_color(theta)
        pygame.draw.line(screen, color, (cart_sx, track_y), (bob_x, bob_y), 4)
        pygame.draw.circle(screen, color, (bob_x, bob_y), 10)

        # ---- Live state panel (top-left) ----
        pygame.draw.rect(screen, PANEL_BG, (10, 10, 320, 120), border_radius=6)
        labels = [
            f"x     = {x_pos:+7.3f} m",
            f"v_x   = {v_x:+7.3f} m/s",
            f"theta = {theta:+7.4f} rad  ({math.degrees(theta):+6.1f} deg)",
            f"omega = {omega:+7.3f} rad/s",
        ]
        for i, txt in enumerate(labels):
            surf = font_small.render(txt, True, TEXT_COLOR)
            screen.blit(surf, (20, 18 + i * 20))

        # Force indicator
        force_label = f"Force = {force:+6.1f} N"
        surf = font_med.render(force_label, True, TEXT_COLOR)
        screen.blit(surf, (20, 100))
        # small bar
        bar_x, bar_y, bar_w = 220, 105, 100
        pygame.draw.rect(screen, (60, 60, 70), (bar_x, bar_y, bar_w, 10))
        if force != 0:
            frac = force / args.force_magnitude
            c = FORCE_BAR_POS if force > 0 else FORCE_BAR_NEG
            bw = int(abs(frac) * bar_w // 2)
            if force > 0:
                pygame.draw.rect(screen, c, (bar_x + bar_w // 2, bar_y, bw, 10))
            else:
                pygame.draw.rect(screen, c, (bar_x + bar_w // 2 - bw, bar_y, bw, 10))

        # Cart position bar
        pos_bar_y = 85
        pygame.draw.rect(screen, (60, 60, 70), (220, pos_bar_y, 100, 6))
        pos_frac = (x_pos + track_limit) / (2 * track_limit)
        pos_frac = max(0.0, min(1.0, pos_frac))
        marker_x = 220 + int(pos_frac * 100)
        pygame.draw.circle(screen, CART_COLOR, (marker_x, pos_bar_y + 3), 5)
        surf = font_small.render("pos", True, DIM_TEXT)
        screen.blit(surf, (195, pos_bar_y - 3))

        # ---- Session stats panel (top-right) ----
        pygame.draw.rect(screen, PANEL_BG,
                         (STATS_PANEL_X, 10, SCREEN_WIDTH - STATS_PANEL_X - 10, 120),
                         border_radius=6)
        total_eps = len(all_episodes)
        total_steps_all = sum(len(ep["forces"]) for ep in all_episodes)
        completed_lengths = [len(ep["forces"]) for ep in all_episodes]
        mean_len = np.mean(completed_lengths) if completed_lengths else 0.0
        elapsed = time.time() - session_start
        mins, secs = divmod(int(elapsed), 60)

        stats = [
            f"Session episodes: {session_completed}",
            f"Total episodes:   {total_eps}",
            f"Current step:     {step}",
            f"Mean ep length:   {mean_len:.0f}",
            f"Total timesteps:  {total_steps_all + step}",
            f"Session time:     {mins:02d}:{secs:02d}",
        ]
        for i, txt in enumerate(stats):
            surf = font_small.render(txt, True, TEXT_COLOR)
            screen.blit(surf, (STATS_PANEL_X + 12, 18 + i * 17))

        # ---- Flash message ----
        if time.time() < flash_until:
            surf = font_med.render(flash_msg, True, flash_color)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, CANVAS_TOP + CANVAS_HEIGHT + 30))
            screen.blit(surf, rect)

        # ---- Control hints bar (bottom) ----
        hints_y = SCREEN_HEIGHT - HINTS_HEIGHT
        pygame.draw.rect(screen, HINT_BG, (0, hints_y, SCREEN_WIDTH, HINTS_HEIGHT))
        hints = ("LEFT: push left  |  RIGHT: push right  |  "
                 "DOWN: zero force  |  R: reset episode  |  "
                 "S: save & quit  |  Q: quit")
        surf = font_small.render(hints, True, DIM_TEXT)
        rect = surf.get_rect(center=(SCREEN_WIDTH // 2, hints_y + HINTS_HEIGHT // 2))
        screen.blit(surf, rect)

        pygame.display.flip()
        clock.tick(args.fps)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    pygame.quit()

    # Final summary
    new_eps = len(all_episodes) - episodes_at_start
    total_new_steps = sum(
        len(ep["forces"]) for ep in all_episodes[episodes_at_start:]
    )
    print(f"\nSession summary:")
    print(f"  New episodes collected: {new_eps}")
    print(f"  New timesteps:          {total_new_steps}")
    print(f"  Total episodes in file: {len(all_episodes)}")
    print(f"  Output: {args.output_path}")


if __name__ == "__main__":
    main()
