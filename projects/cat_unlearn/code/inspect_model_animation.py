# sr_sim_and_anim_refactor.py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # render offscreen for pygame
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pygame as pg

# assumes you provide make_stim_cats() in util_func.py
from util_func import make_stim_cats


# ------------------------------------------------------------
# 1) SIMULATION
# ------------------------------------------------------------
def simulate_model(
    n_trials=100,
    grid_size=100,
    vis_dim=100.0,
    vis_amp=1.0,
    vis_width=5.0,
    alpha=1e-3,
    alpha_pr=0.05,
    pr_init=0.0,
    rng_seed=0,
):
    """
    Runs the full model once and returns a dict with per-trial records, including
    W_pre and W_post so the animation can step back/forward correctly.
    """
    rng = np.random.default_rng(rng_seed)

    # Stimuli
    d = make_stim_cats()
    # If caller wants fewer trials than d, slice; else reuse modulo
    if n_trials > len(d):
        # tile the dataset if needed
        reps = int(np.ceil(n_trials / len(d)))
        d = pd.concat([d] * reps, ignore_index=True)
    d = d.iloc[:n_trials].reset_index(drop=True)

    stim_x = d["x"].to_numpy(np.float32)
    stim_y = d["y"].to_numpy(np.float32)
    stim_cat = d["cat"].to_numpy()

    # Visual grid
    xs = np.linspace(0.0, vis_dim, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    n_vis = grid_size * grid_size

    # Weights init in [0, 0.5)
    W = rng.random((n_vis, 2), dtype=np.float32) * 0.5

    # State trackers
    resp = np.empty(n_trials, dtype="<U1")
    r = np.zeros(n_trials, dtype=np.float32)
    pr = np.zeros(n_trials + 1,
                  dtype=np.float32)
    rpe = np.zeros(n_trials, dtype=np.float32)
    pr[0] = pr_init

    # Per-trial saved arrays
    vis_acts = []  # (grid, grid) per trial
    out_pre = np.zeros((n_trials, 2), dtype=np.float32)
    out_post = np.zeros((n_trials, 2), dtype=np.float32)
    W_preA = np.zeros((n_trials, grid_size, grid_size), dtype=np.float32)
    W_preB = np.zeros((n_trials, grid_size, grid_size), dtype=np.float32)
    W_postA = np.zeros((n_trials, grid_size, grid_size), dtype=np.float32)
    W_postB = np.zeros((n_trials, grid_size, grid_size), dtype=np.float32)

    for t in range(n_trials):
        # Save W pre-update
        W_preA[t] = W[:, 0].reshape(grid_size, grid_size)
        W_preB[t] = W[:, 1].reshape(grid_size, grid_size)

        x, y, cat = stim_x[t], stim_y[t], stim_cat[t]

        # Visual activation
        dx2 = (X - x)**2
        dy2 = (Y - y)**2
        vis_act = vis_amp * np.exp(-(dx2 + dy2) / (2.0 * (vis_width**2)))
        vis_act = vis_act.astype(np.float32)
        vis_acts.append(vis_act)

        phi = vis_act.ravel()
        out = phi @ W  # pre-WTA
        out_pre[t] = out

        # Greedy choice
        resp[t] = "B" if out[1] > out[0] else "A"

        # Post-WTA bars (inhibition)
        out_post_t = out.copy()
        if resp[t] == "A":
            out_post_t[1] = 0.0
        else:
            out_post_t[0] = 0.0
        out_post[t] = out_post_t

        # Reward & RPE
        r[t] = 1.0 if resp[t] == cat else -1.0
        rpe[t] = r[t] - pr[t]
        pr[t + 1] = pr[t] + alpha_pr * rpe[t]

        # Weight update (AFTER feedback in animation, but here we precompute the post-update)
        if rpe[t] > 0:
            W[:, 0] += alpha * rpe[t] * phi * out_post_t[0] * W[:, 0]
            W[:, 1] += alpha * rpe[t] * phi * out_post_t[1] * W[:, 1]
        elif rpe[t] < 0:
            W[:, 0] += alpha * rpe[t] * phi * out_post_t[0] * (1.0 - W[:, 0])
            W[:, 1] += alpha * rpe[t] * phi * out_post_t[1] * (1.0 - W[:, 1])

        # Save W post-update
        W_postA[t] = W[:, 0].reshape(grid_size, grid_size)
        W_postB[t] = W[:, 1].reshape(grid_size, grid_size)

    # Pack results
    sim = {
        "n_trials": n_trials,
        "grid_size": grid_size,
        "vis_dim": vis_dim,
        "vis_width": vis_width,
        "stim_x": stim_x,
        "stim_y": stim_y,
        "stim_cat": stim_cat,
        "resp": resp,
        "r": r,
        "pr": pr[:-1],  # pr at trial t (length n_trials)
        "rpe": rpe,
        "vis_acts": vis_acts,
        "out_pre": out_pre,
        "out_post": out_post,
        "W_preA": W_preA,
        "W_preB": W_preB,
        "W_postA": W_postA,
        "W_postB": W_postB,
    }
    return sim


# ------------------------------------------------------------
# 2) ANIMATION
# ------------------------------------------------------------
def animate_simulation(sim, speed=1.0):
    """
    Plays the animation with play/pause (space) and event stepping (←/→).
    Uses precomputed W_pre/W_post per trial so stepping shows the correct weights.
    """
    # Unpack
    n_trials = sim["n_trials"]
    grid_size = sim["grid_size"]
    vis_dim = sim["vis_dim"]

    stim_x = sim["stim_x"]
    stim_y = sim["stim_y"]
    stim_cat = sim["stim_cat"]
    resp = sim["resp"]
    r = sim["r"]
    pr = sim["pr"]
    rpe = sim["rpe"]
    vis_acts = sim["vis_acts"]
    out_pre = sim["out_pre"]
    out_post = sim["out_post"]
    W_preA = sim["W_preA"]
    W_preB = sim["W_preB"]
    W_postA = sim["W_postA"]
    W_postB = sim["W_postB"]

    # --- Pygame / figure UI ---
    pg.init()
    pg.display.set_caption(
        "SR Category Learning — (Simulated) Square Panels + Controls")
    WIN_W, WIN_H = 1280, 720
    screen = pg.display.set_mode((WIN_W, WIN_H),
                                 flags=pg.HWSURFACE | pg.DOUBLEBUF)
    clock = pg.time.Clock()
    FONT = pg.font.SysFont("Arial", 18)
    SCALE = 0.95

    # --- Timing (ms), apply speed factor ---
    def ms(v):
        return max(16, int(v * speed))

    BLANK_MS = ms(300)  # pre-fix blank
    FIX_MS = ms(400)
    RT_MS = ms(700)  # stim + PRE-WTA bars
    WTA_MS = ms(300)  # POST-WTA bars
    FB_MS = ms(500)  # feedback halo
    POSTBLANK_MS = ms(300)  # ITI

    EVENTS = [
        "BLANK", "FIX", "STIM_PRE", "STIM_POST", "FEEDBACK", "POST_BLANK"
    ]
    DURATION = [BLANK_MS, FIX_MS, RT_MS, WTA_MS, FB_MS, POSTBLANK_MS]

    # --- Grating rendering setup ---
    stim_w, stim_h = 480, 480
    spatial_freq = 8
    phase = 0.0  # static
    yy_stim, xx_stim = np.mgrid[0:stim_h, 0:stim_w]
    cx, cy = stim_w / 2.0, stim_h / 2.0
    radius = 0.44 * min(stim_w, stim_h)
    ap_mask = ((xx_stim - cx)**2 + (yy_stim - cy)**2) <= radius**2
    bg = 0.5

    # --- Figure builder with vis/resp toggles ---
    def make_figure(trial_idx, event_name):
        y_for_ori = stim_y[trial_idx]
        vis_act_img = vis_acts[trial_idx]
        # choose which weights to show
        if event_name in ("POST_BLANK", ):
            WA = W_postA[trial_idx]
            WB = W_postB[trial_idx]
        else:
            WA = W_preA[trial_idx]
            WB = W_preB[trial_idx]

        bars_pre = out_pre[trial_idx]
        bars_post = out_post[trial_idx]
        show_post = event_name in ("STIM_POST", "FEEDBACK")
        show_grating = event_name in ("STIM_PRE", "STIM_POST", "FEEDBACK")
        show_vis = show_grating
        show_resp = show_grating
        halo_color = ("limegreen" if r[trial_idx] > 0 else
                      "red") if event_name == "FEEDBACK" else None

        fig_w = int(SCALE * WIN_W)
        fig_h = int(SCALE * WIN_H)
        fig = plt.figure(figsize=(fig_w / 100.0, fig_h / 100.0),
                         dpi=100,
                         constrained_layout=True)

        gs = GridSpec(3,
                      5,
                      figure=fig,
                      height_ratios=[1.0, 1.0, 0.9],
                      width_ratios=[1.1, 1.1, 1.1, 1.1, 0.35])

        # containers
        ax_gratC = fig.add_subplot(gs[0:2, 0])
        ax_visC = fig.add_subplot(gs[0:2, 1])
        ax_WA = fig.add_subplot(gs[0, 2])
        ax_WB = fig.add_subplot(gs[1, 2])
        ax_respC = fig.add_subplot(gs[0:2, 3])
        ax_lines = fig.add_subplot(gs[2, 0:4])

        for ax in (ax_gratC, ax_visC, ax_respC):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        # weights (clamped display in [0,1])
        ax_WA.imshow(np.clip(WA, 0, 1),
                     origin='lower',
                     cmap='magma',
                     vmin=0,
                     vmax=1,
                     interpolation='nearest')
        ax_WA.set_title("Weights → A", fontsize=11)
        ax_WA.set_xticks([])
        ax_WA.set_yticks([])
        ax_WB.imshow(np.clip(WB, 0, 1),
                     origin='lower',
                     cmap='magma',
                     vmin=0,
                     vmax=1,
                     interpolation='nearest')
        ax_WB.set_title("Weights → B", fontsize=11)
        ax_WB.set_xticks([])
        ax_WB.set_yticks([])

        # square sizing based on weight axis height
        fig.canvas.draw_idle()
        bbox_WA = ax_WA.get_position()
        bbox_g = ax_gratC.get_position()
        bbox_v = ax_visC.get_position()
        bbox_r = ax_respC.get_position()
        sq_h = bbox_WA.height
        sq_w = sq_h

        def add_square(bbox, title=None):
            x0 = bbox.x0 + (bbox.width - sq_w) / 2.0
            y0 = bbox.y0 + (bbox.height - sq_h) / 2.0
            ax = fig.add_axes([x0, y0, sq_w, sq_h])
            if title: ax.set_title(title, fontsize=11)
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            return ax

        ax_grat = add_square(bbox_g, title="Stimulus")
        ax_vis = add_square(bbox_v, title="Visual Activation")
        title_bars = "Response Activations (POST-WTA)" if show_post else "Response Activations (PRE-WTA)"
        ax_resp = add_square(bbox_r, title=title_bars)

        # stimulus or fixation/blank
        if show_grating:
            theta = np.deg2rad((y_for_ori / vis_dim) * 180.0)
            xr = (xx_stim - cx) * np.cos(theta) + (yy_stim -
                                                   cy) * np.sin(theta)
            g = np.sin(2 * np.pi * spatial_freq * xr / stim_w + phase)
            img = np.full((stim_h, stim_w), bg, dtype=np.float32)
            img[ap_mask] = 0.5 + 0.5 * g[ap_mask]
            ax_grat.imshow(img,
                           cmap='gray',
                           vmin=0,
                           vmax=1,
                           origin='lower',
                           interpolation='nearest')
            if halo_color is not None:
                halo = Circle((stim_w / 2.0, stim_h / 2.0),
                              radius * 1.03,
                              edgecolor=halo_color,
                              facecolor='none',
                              linewidth=3)
                ax_grat.add_patch(halo)
        else:
            ax_grat.set_xlim(0, stim_w)
            ax_grat.set_ylim(0, stim_h)
            ax_grat.set_aspect('equal', adjustable='box')
            ax_grat.set_facecolor((0.5, 0.5, 0.5))
            if event_name == "FIX":
                size = 0.08 * min(stim_w, stim_h)
                ax_grat.plot([stim_w / 2 - size, stim_w / 2 + size],
                             [stim_h / 2, stim_h / 2],
                             color='k',
                             lw=3)
                ax_grat.plot([stim_w / 2, stim_w / 2],
                             [stim_h / 2 - size, stim_h / 2 + size],
                             color='k',
                             lw=3)

        # visual activation only when stimulus visible
        if show_vis:
            ax_vis.imshow(np.clip(vis_act_img, 0, 1),
                          origin='lower',
                          cmap='viridis',
                          vmin=0,
                          vmax=1,
                          interpolation='nearest')
        else:
            ax_vis.set_facecolor((0.2, 0.2, 0.2))

        # response bars only when stimulus visible
        if show_resp:
            vals_pre = [
                float(out_pre[trial_idx, 0]),
                float(out_pre[trial_idx, 1])
            ]
            vals_post = [
                float(out_post[trial_idx, 0]),
                float(out_post[trial_idx, 1])
            ]
            draw_vals = vals_post if show_post else vals_pre
            ax_resp.bar(["A", "B"], draw_vals)
            ymax = max(1e-6, float(max(vals_pre + vals_post)) * 1.05)
            ax_resp.set_ylim(0, ymax)
        else:
            ax_resp.set_facecolor((0.2, 0.2, 0.2))

        # reward plot ([-1,1], 1..n_trials)
        # number of completed trials for the plot = trials that have passed FEEDBACK
        completed = trial_idx  # default
        if event_name in ("POST_BLANK", ):
            completed = trial_idx + 1  # after feedback, we consider t completed

        if completed > 0:
            xs_line = np.arange(1, completed + 1)
            ax_lines.plot(xs_line, r[:completed], label="r")
            ax_lines.plot(xs_line, pr[:completed], label="pr")
            ax_lines.plot(xs_line, rpe[:completed], label="rpe")
        ax_lines.set_xlim(1, n_trials)
        ax_lines.set_ylim(-1.15, 1.15)
        step = max(1, n_trials // 10)
        ax_lines.set_xticks(np.arange(1, n_trials + 1, step))
        ax_lines.set_title("Reward plot (r, pr, rpe)", fontsize=11)
        ax_lines.legend(fontsize=9, loc="upper right", frameon=False)

        return fig

    # ---- event engine ----
    running = True
    trial = 0
    event_idx = 0
    phase_start_ms = pg.time.get_ticks()
    paused = False

    while running and trial < n_trials:
        now = pg.time.get_ticks()

        # input
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_SPACE:
                    paused = not paused
                elif ev.key == pg.K_RIGHT:
                    # advance event
                    event_idx += 1
                    if event_idx >= len(EVENTS):
                        event_idx = 0
                        trial = min(trial + 1, n_trials - 1)
                    phase_start_ms = pg.time.get_ticks()
                elif ev.key == pg.K_LEFT:
                    # retreat event
                    if event_idx > 0:
                        event_idx -= 1
                    else:
                        if trial > 0:
                            trial -= 1
                            event_idx = len(EVENTS) - 1
                    phase_start_ms = pg.time.get_ticks()

        # auto advance
        if not paused:
            elapsed = now - phase_start_ms
            if elapsed >= DURATION[event_idx]:
                event_idx += 1
                if event_idx >= len(EVENTS):
                    event_idx = 0
                    trial += 1
                    if trial >= n_trials:
                        trial = n_trials - 1
                        running = False
                        continue
                phase_start_ms = pg.time.get_ticks()

        # build & draw current frame
        evname = EVENTS[event_idx]
        fig = make_figure(trial, evname)
        canvas = FigureCanvas(fig)
        canvas.draw()
        raw = canvas.buffer_rgba()
        fw, fh = canvas.get_width_height()
        surf = pg.image.frombuffer(raw, (fw, fh), "RGBA").convert_alpha()
        plt.close(fig)

        screen.fill((20, 20, 26))
        screen.blit(surf, ((WIN_W - fw) // 2, (WIN_H - fh) // 2))

        status = f"{evname} — Trial {trial+1}/{n_trials} | resp={resp[trial]} | r={r[trial]:+.0f} | rpe={rpe[trial]:+.3f} | {'PAUSED' if paused else 'PLAY'}"
        screen.blit(FONT.render(status, True, (235, 235, 235)),
                    (30, WIN_H - 36))
        pg.display.flip()
        clock.tick(30)

    pg.quit()


# ------------------------------------------------------------
# 3) Run both
# ------------------------------------------------------------
if __name__ == "__main__":
    sim_rec = simulate_model(
        n_trials=100,
        grid_size=100,
        vis_dim=100.0,
        vis_amp=1.0,
        vis_width=5.0,
        alpha=5e-3,
        alpha_pr=5e-2,
        pr_init=0.0,
        rng_seed=0,
    )
    # speed < 1.0 = faster (e.g., 0.5 = 2x faster). speed > 1.0 = slower.
    animate_simulation(sim_rec, speed=0.01)
