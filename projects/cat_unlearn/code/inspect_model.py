import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util_func import *

# generate stimuli and categories
d = make_stim_cats()

# model / sim params (defined once)
grid_size   = 100                 # 100 x 100 grid of visual units
vis_dim     = 100.0               # stimuli live in [0, 100] x [0, 100]
vis_amp     = 1.0                 # RBF amplitude
vis_width   = 10.0                # RBF width (sigma)
alpha       = 1e-2                # learning rate for weights
alpha_pr    = 0.01                # learning rate for predicted reward
pr_init     = 0.0                 # initial predicted reward
n_out       = 2                   # two category outputs: A, B
rng         = np.random.default_rng(0)

# n_trials = d.shape[0]
n_trials = 100

# precompute static visual grid once
xs = np.linspace(0.0, vis_dim, grid_size)
X, Y = np.meshgrid(xs, xs, indexing='xy')
n_vis = grid_size * grid_size

# initialize weights once (small random numbers in [0, 0.01])
W = rng.random((n_vis, n_out)) * 0.01

# pull stimulus arrays for fast NumPy indexing
stim_x = d['x'].to_numpy()
stim_y = d['y'].to_numpy()
stim_cat = d['cat'].to_numpy()

# predicted reward state (carried across trials)
resp = np.zeros(n_trials, dtype='<U1')
r = np.zeros(n_trials)
pr = np.zeros(n_trials)
rpe = np.zeros(n_trials)
pr[0] = pr_init

# trial loop
for trl in range(n_trials-1):
    # get stimulus coordinates and category
    x = stim_x[trl]
    y = stim_y[trl]
    cat = stim_cat[trl]

    # compute visual unit activations (RBF centered at (x, y))
    # vis_act shape: (grid_size, grid_size)
    dx2 = (X - x) ** 2
    dy2 = (Y - y) ** 2
    vis_act = vis_amp * np.exp(-(dx2 + dy2) / (2.0 * vis_width ** 2))

    # flatten once (view) for dot products / updates
    phi = vis_act.ravel()  # shape (n_vis,)

    # compute output unit activations
    # out_act: (2,)
    out_act = phi @ W

    # greedy action selection
    resp[trl] = 'B' if out_act[1] > out_act[0] else 'A'

    # strong lateral inhibition between output units (zero the loser)
    if resp[trl] == 'A':
        out_act[1] = 0.0
    else:
        out_act[0] = 0.0

    # determine reward based on true category
    r[trl] = 1 if resp[trl] == cat else -1

    # compute RPE and update predicted reward (SR-style scalar critic)
    rpe[trl] = r[trl] - pr[trl]
    pr[trl+1] = alpha_pr * rpe[trl]

    # update weights according to reinforcement learning rule
    if rpe[trl] > 0:
        W[:, 0] += alpha * rpe[trl] * phi * out_act[0] * W[:, 0]
        W[:, 1] += alpha * rpe[trl] * phi * out_act[1] * W[:, 1]
    elif rpe[trl] < 0:
        W[:, 0] += alpha * rpe[trl] * phi * out_act[0] * (1 - W[:, 0])
        W[:, 1] += alpha * rpe[trl] * phi * out_act[1] * (1 - W[:, 1])

    # plot vis_act, W[:, 0], W[:, 1] as images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.title('Visual Unit Activations')
    plt.imshow(vis_act, origin='lower')

    plt.subplot(1, 3, 2)
    plt.title('Weights to Category A')
    plt.imshow(W[:, 0].reshape(grid_size, grid_size), origin='lower')

    plt.subplot(1, 3, 3)
    plt.title('Weights to Category B')
    plt.imshow(W[:, 1].reshape(grid_size, grid_size), origin='lower')

    plt.tight_layout()
    plt.show()

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(8, 6))
sns.lineplot(x=np.arange(n_trials), y=pr[:n_trials], ax=ax[0, 0])
sns.lineplot(x=np.arange(n_trials), y=rpe[:n_trials], ax=ax[0, 1])
ax[0, 0].set_xticks(np.arange(0, n_trials, 10))
ax[0, 1].set_xticks(np.arange(0, n_trials, 10))
plt.tight_layout()
plt.show()
