from imports import *
from util_func import make_stim_cats, create_grating_patch
import matplotlib.pyplot as plt


def make_example_trials_figure(
        save_path="../figures/example_trials_figure.png", dpi=300):

    # ---------------------------------------
    # 1. Generate stim dataset and pick trials
    # ---------------------------------------
    ds = make_stim_cats()

    # Pick one A example and one B example
    trial1 = ds[ds.cat == "A"].iloc[0]
    trial2 = ds[ds.cat == "B"].iloc[1]

    # ---------------------------------------
    # 2. Display geometry
    # ---------------------------------------
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 3
    size_px = int(size_cm * px_per_cm)

    def stim_to_grating(row):
        freq = row["xt"] * (px_per_cm**-1)
        theta = row["yt"]
        return create_grating_patch(size_px, freq, theta)

    g1 = stim_to_grating(trial1)
    g2 = stim_to_grating(trial2)

    # ---------------------------------------
    # 3. Drawing helpers
    # ---------------------------------------
    def draw_fixation(ax):
        ax.set_facecolor((0.5, 0.5, 0.5))
        ax.plot([0, 0], [-0.15, 0.15], linewidth=4, color="white")
        ax.plot([-0.15, 0.15], [0, 0], linewidth=4, color="white")

        # slightly larger limits so they match the stimulus/feedback panels
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    def draw_grating(ax, grating, text=None):
        ax.set_facecolor((0.5, 0.5, 0.5))

        # grating itself still lives in [-0.5, 0.5] -> appears slightly smaller
        ax.imshow(grating,
                  cmap="gray",
                  interpolation="nearest",
                  extent=(-0.5, 0.5, -0.5, 0.5))

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

        if text is not None:
            ax.text(0,
                    -0.7,
                    text,
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=9)

    def draw_feedback(ax, grating, feedback):
        ax.set_facecolor((0.5, 0.5, 0.5))

        ax.imshow(grating,
                  cmap="gray",
                  interpolation="nearest",
                  extent=(-0.5, 0.5, -0.5, 0.5))

        # expanded limits so the halo isn't clipped
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

        color = "lime" if feedback == "Correct" else "red"
        circle = plt.Circle(
            (0, 0),
            0.55,  # same radius as before
            fill=False,
            linewidth=4,
            edgecolor=color)
        ax.add_patch(circle)

        ax.text(0,
                -0.7,
                feedback,
                ha="center",
                va="top",
                color="white",
                fontsize=9)

    # ---------------------------------------
    # 4. Layout: diagonal / overlapping timeline
    # ---------------------------------------
    fig = plt.figure(figsize=(14, 8))

    # base positions (x,y) in figure coordinates
    # slightly overlapping successive steps (delta < panel width)
    panel_w = 0.24
    panel_h = 0.24
    dx = 0.1  # overlap horizontally
    dy = 0.20  # stagger downwards

    # Trial 1 positions
    pos_T1_iti = [0.05, 0.70, panel_w, panel_h]
    pos_T1_stim = [0.05 + dx, 0.70 - dy, panel_w, panel_h]
    pos_T1_fb = [0.05 + 2 * dx, 0.70 - 2 * dy, panel_w, panel_h]

    # Trial 2 positions (shift down further)
    pos_T2_iti = [0.5, 0.70, panel_w, panel_h]
    pos_T2_stim = [0.5 + dx, 0.70 - dy, panel_w, panel_h]
    pos_T2_fb = [0.5 + 2 * dx, 0.70 - 2 * dy, panel_w, panel_h]

    # -----------------------
    # Trial 1
    # -----------------------
    ax = fig.add_axes(pos_T1_iti)
    draw_fixation(ax)
    ax.set_title("", fontsize=9)

    ax = fig.add_axes(pos_T1_stim)
    draw_grating(ax, g1, text="Press D for A\nPress K for B")
    ax.set_title("", fontsize=9)

    ax = fig.add_axes(pos_T1_fb)
    draw_feedback(ax, g1, "Correct")
    ax.set_title("", fontsize=9)

    # -----------------------
    # Trial 2
    # -----------------------
    ax = fig.add_axes(pos_T2_iti)
    draw_fixation(ax)
    ax.set_title("", fontsize=9)

    ax = fig.add_axes(pos_T2_stim)
    draw_grating(ax, g2, text="Press D for A\nPress K for B")
    ax.set_title("", fontsize=9)

    ax = fig.add_axes(pos_T2_fb)
    draw_feedback(ax, g2, "Incorrect")
    ax.set_title("", fontsize=9)

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved updated overlapped-flow figure as '{save_path}'")


if __name__ == "__main__":
    make_example_trials_figure()
