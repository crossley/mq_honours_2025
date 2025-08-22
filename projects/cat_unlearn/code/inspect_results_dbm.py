import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pingouin as pg
from util_func_dbm import *

if __name__ == "__main__":

    dir_data = "../data"

    d_rec = []

    for file in os.listdir(dir_data):

        if file.endswith(".csv"):
            d = pd.read_csv(os.path.join(dir_data, file))
            d["phase"] = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"
                                                                     ] * 299
            d_rec.append(d)

    d = pd.concat(d_rec, ignore_index=True)

    d.groupby(["experiment", "condition"])["subject"].unique()
    d.groupby(["experiment", "condition"])["subject"].nunique()

    d.loc[d["cat"] == "A", "cat"] = 0
    d.loc[d["cat"] == "B", "cat"] = 1
    d.loc[d["resp"] == "A", "resp"] = 0
    d.loc[d["resp"] == "B", "resp"] = 1
    d["cat"] = d["cat"].astype(int)
    d["resp"] = d["resp"].astype(int)
    d["acc"] = d["cat"] == d["resp"]

    # TODO: annoying fix this
    block_size = 25
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size

    # define exc_subs to be numpy array of subjects that did not reach greater than 65% accuracy
    # during bloack 2
    dd = d.groupby(["experiment", "condition", "subject", "block"])["acc"].mean().reset_index()
    exc_subs = dd[(dd["block"] == 2) & (dd["acc"] < 0.6)]["subject"].unique()
    d = d[~d["subject"].isin(exc_subs)]

    # NOTE: focus on the last learning block and the test blocks
    d = d.loc[(d["block"] == 2) | (d["block"] == 6)]

    d = d.sort_values(["experiment", "condition", "subject", "block", "trial"])

    models = [
        nll_unix,
        nll_unix,
        nll_uniy,
        nll_uniy,
        nll_glc,
        nll_glc,
        #    nll_gcc_eq,
        #    nll_gcc_eq,
        #    nll_gcc_eq,
        #    nll_gcc_eq,
    ]
    side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
    k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    n = block_size
    model_names = [
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
        "nll_glc_0",
        "nll_glc_1",
        #    "nll_gcc_eq_0",
        #    "nll_gcc_eq_1",
        #    "nll_gcc_eq_2",
        #    "nll_gcc_eq_3",
    ]

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        best_model = np.unique(model[bic == bic.min()])[0]
        x["best_model"] = best_model
        return x

    if not os.path.exists("../dbm_fits/dbm_results.csv"):
        dbm = (d.groupby(["experiment", "condition", "subject",
                          "block"]).apply(fit_dbm, models, side, k, n,
                                          model_names).reset_index())

        dbm.to_csv("../dbm_fits/dbm_results.csv")

    else:
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")

    dbm = dbm.groupby(["experiment", "condition", "subject",
                       "block"]).apply(assign_best_model)

    # exclude subjects that did not reach greater than 65% accuracy
    dbm = dbm[~dbm["subject"].isin(exc_subs)]

    dd = dbm.loc[dbm["model"] == dbm["best_model"]]

    ddd = dd[["experiment", "condition", "subject", "block",
              "best_model"]].drop_duplicates()
    ddd["best_model_class"] = ddd["best_model"].str.split("_").str[1]
    ddd.loc[ddd["best_model_class"] != "glc",
            "best_model_class"] = "rule-based"
    ddd.loc[ddd["best_model_class"] == "glc",
            "best_model_class"] = "procedural"
    ddd["best_model_class"] = ddd["best_model_class"].astype("category")
    ddd = ddd.reset_index(drop=True)

    def get_best_model_class_2(x):
        if np.isin("rule-based", x["best_model_class"].to_numpy()):
            x["best_model_class_2"] = "rule-based"
        else:
            x["best_model_class_2"] = "procedural"

        return x

    ddd = ddd.groupby(["experiment", "condition", "block", "subject"
                       ]).apply(get_best_model_class_2).reset_index(drop=True)
    ddd["best_model_class_2"] = ddd["best_model_class_2"].astype("category")
    ddd['block'] = ddd['block'].astype("category")

    # TODO: Clean up plot code etc
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 6))

    CLASS_ORDER = ["procedural", "rule-based"]

    def block_cross_counts(ddd, experiment, condition, b_y=2, b_x=6):
        d_y = ddd[(ddd["experiment"] == experiment)
                  & (ddd["condition"] == condition) & (ddd["block"] == b_y)][[
                      "subject", "best_model_class_2"
                  ]].rename(columns={"best_model_class_2": f"b{b_y}"})
        d_x = ddd[(ddd["experiment"] == experiment)
                  & (ddd["condition"] == condition) & (ddd["block"] == b_x)][[
                      "subject", "best_model_class_2"
                  ]].rename(columns={"best_model_class_2": f"b{b_x}"})

        # only subjects present in both blocks
        both = pd.merge(d_y, d_x, on="subject", how="inner")
        if both.empty:
            return pd.DataFrame(0, index=CLASS_ORDER, columns=CLASS_ORDER)

        # enforce ordering
        both[f"b{b_y}"] = pd.Categorical(both[f"b{b_y}"],
                                         categories=CLASS_ORDER)
        both[f"b{b_x}"] = pd.Categorical(both[f"b{b_x}"],
                                         categories=CLASS_ORDER)

        ct = pd.crosstab(both[f"b{b_y}"],
                         both[f"b{b_x}"]).reindex(index=CLASS_ORDER,
                                                  columns=CLASS_ORDER,
                                                  fill_value=0)
        return ct

    def draw_heatmap(ax, counts, title, xlabel="Block 6", ylabel="Block 2"):
        im = ax.imshow(counts.values, aspect="equal", cmap="Blues")
        # ticks & labels
        ax.set_xticks(range(len(counts.columns)))
        ax.set_yticks(range(len(counts.index)))
        ax.set_xticklabels(counts.columns, rotation=0)
        ax.set_yticklabels(counts.index)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # annotate cell counts
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                ax.text(j, i, str(counts.iat[i, j]), ha="center", va="center")

        # add a small colorbar for this axes
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (row, col, experiment, condition, title)
    panels = [
        (0, 0, 1, "relearn", "Exp 1 — relearn"),
        (0, 1, 1, "new_learn", "Exp 1 — new_learn"),
        (1, 0, 2, "relearn", "Exp 2 — relearn"),
        (1, 1, 2, "new_learn", "Exp 2 — new_learn"),
    ]

    for r, c, exp, cond, title in panels:
        counts = block_cross_counts(ddd, exp, cond, b_y=2, b_x=6)
        draw_heatmap(ax[r, c], counts, title)

    fig.suptitle("Best model class: Block 2 (Y) vs Block 6 (X)", y=1.02)
    fig.tight_layout()
    plt.savefig("../figures/best_model_class_heatmap.png", dpi=300)
    plt.close()

    # NOTE: examine only procedural subjects
    proc_subs_2 = ddd[(ddd["best_model_class_2"] == "procedural")
                      & (ddd["block"] == 2)]["subject"].unique()
    proc_subs_6 = ddd[(ddd["best_model_class_2"] == "procedural")
                      & (ddd["block"] == 6)]["subject"].unique()
    rb_subs_2 = ddd[(ddd["best_model_class_2"] == "rule-based")
                    & (ddd["block"] == 2)]["subject"].unique()
    rb_subs_6 = ddd[(ddd["best_model_class_2"] == "rule-based")
                    & (ddd["block"] == 6)]["subject"].unique()
    proc_proc_subs = np.intersect1d(proc_subs_2, proc_subs_6)
    proc_rb_subs = np.intersect1d(proc_subs_2, rb_subs_6)

    block_size = 25
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size

    dp = d[d["subject"].isin(proc_proc_subs)]

    dp.groupby(["experiment", "condition"])["subject"].nunique()

    dp["condition"] = dp["condition"].astype("category")
    dp1 = dp[dp["experiment"] == 1]
    dp2 = dp[dp["experiment"] == 2]

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    sns.lineplot(data=dp1, x="block", y="acc", hue="condition", ax=ax[0, 0])
    sns.lineplot(data=dp2, x="block", y="acc", hue="condition", ax=ax[0, 1])
    ax[0, 0].set_title("Experiment 1")
    ax[0, 1].set_title("Experiment 2")
    plt.suptitle("Accuracy of procedural subjects")
    plt.tight_layout()
    plt.savefig("../figures/subjects_accuracy_proc.png", dpi=300)
    plt.close()

    dpp = d.copy()
    dp = dpp.groupby(["experiment", "condition", "subject", "block"])["acc"].mean().reset_index()

    dp.groupby(["experiment", "condition"])["subject"].nunique()

    dp["condition"] = dp["condition"].astype("category")
    dp1 = dp[dp["experiment"] == 1]
    dp2 = dp[dp["experiment"] == 2]

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
    sns.lineplot(data=dp1, x="block", y="acc", hue="condition", ax=ax[0, 0])
    sns.lineplot(data=dp2, x="block", y="acc", hue="condition", ax=ax[0, 1])
    ax[0, 0].set_title("Experiment 1")
    ax[0, 1].set_title("Experiment 2")
    plt.suptitle("Accuracy of all subjects")
    plt.tight_layout()
    plt.savefig("../figures/subjects_accuracy_all.png", dpi=300)
    plt.close()

###################
# NOTE: Bayes stats
###################

def get_posterior_diff(exp1_success,
                       exp1_total,
                       exp2_success,
                       exp2_total,
                       samples=100000):
    theta1 = np.random.beta(exp1_success + 1, exp1_total - exp1_success + 1, samples)
    theta2 = np.random.beta(exp2_success + 1, exp2_total - exp2_success + 1, samples)
    delta = theta1 - theta2
    return theta1, theta2, delta

def plot_bayesian_comparison(theta1, theta2, delta, row, axs, condition_label):
    ci = np.percentile(delta, [2.5, 97.5])
    prob_exp1_greater = (delta > 0).mean()

    axs[row, 0].hist(theta1, bins=100, color='gray', density=True)
    axs[row, 0].set_title(
        f'Exp 1 ({condition_label}): Probability of reacquiring\nprocedural strategy (θ₁)',
        fontsize=11)
    axs[row, 0].set_xlabel('θ₁')
    axs[row, 0].set_ylabel('Posterior Probability Density')

    axs[row, 1].hist(theta2, bins=100, color='gray', density=True)
    axs[row, 1].set_title(
        f'Exp 2 ({condition_label}): Probability of reacquiring\nprocedural strategy (θ₂)',
        fontsize=11)
    axs[row, 1].set_xlabel('θ₂')
    axs[row, 1].set_ylabel('Posterior Probability Density')

    axs[row, 2].hist(delta, bins=100, color='gray', density=True)
    axs[row, 2].axvline(0, color='black', linestyle='--', label='Δ = 0')
    axs[row, 2].axvline(ci[0], color='red', linestyle=':', label='95% CI')
    axs[row, 2].axvline(ci[1], color='red', linestyle=':')
    axs[row, 2].set_title(
        f'Difference in reacquisition probability:\nExp 1 − Exp 2 (Δ) [{condition_label}]',
        fontsize=11)
    axs[row, 2].set_xlabel('Δ')
    axs[row, 2].set_ylabel('Posterior Probability Density')
    axs[row, 2].legend()

    print(f"{condition_label} — 95% CI for Δ (Exp 1 − Exp 2): {ci[0]:.3f} to {ci[1]:.3f}")
    print(f"{condition_label} — P(θ₁ > θ₂) = {prob_exp1_greater:.3f}")

# Posterior samples
theta1_relearn, theta2_relearn, delta_relearn = get_posterior_diff(13, 14, 10, 14)
theta1_new, theta2_new, delta_new = get_posterior_diff(6, 18, 5, 13)

# Cross-condition comparisons within experiments
delta_exp1 = theta1_relearn - theta1_new
delta_exp2 = theta2_relearn - theta2_new

# Credible intervals and probabilities
ci_exp1 = np.percentile(delta_exp1, [2.5, 97.5])
ci_exp2 = np.percentile(delta_exp2, [2.5, 97.5])
prob_exp1 = (delta_exp1 > 0).mean()
prob_exp2 = (delta_exp2 > 0).mean()

# Create the figure grid
fig, axs = plt.subplots(3, 3, figsize=(15, 8))
plt.subplots_adjust(hspace=0.75, wspace=0.26)

# Row 1: Relearn condition
plot_bayesian_comparison(theta1_relearn,
                         theta2_relearn,
                         delta_relearn,
                         row=0,
                         axs=axs,
                         condition_label='Relearn')

# Row 2: New Learn condition
plot_bayesian_comparison(theta1_new,
                         theta2_new,
                         delta_new,
                         row=1,
                         axs=axs,
                         condition_label='New Learn')

# Row 3: Relearn − New Learn within each experiment
axs[2, 0].hist(delta_exp1, bins=100, color='gray', density=True)
axs[2, 0].axvline(0, color='black', linestyle='--')
axs[2, 0].axvline(ci_exp1[0], color='red', linestyle=':')
axs[2, 0].axvline(ci_exp1[1], color='red', linestyle=':')
axs[2, 0].set_title(
    'Exp 1: Difference in reacquisition probability\n(Relearn − New Learn)',
    fontsize=11)
axs[2, 0].set_xlabel('Δ')
axs[2, 0].set_ylabel('Posterior Probability Density')

axs[2, 1].hist(delta_exp2, bins=100, color='gray', density=True)
axs[2, 1].axvline(0, color='black', linestyle='--')
axs[2, 1].axvline(ci_exp2[0], color='red', linestyle=':')
axs[2, 1].axvline(ci_exp2[1], color='red', linestyle=':')
axs[2, 1].set_title(
    'Exp 2: Difference in reacquisition probability\n(Relearn − New Learn)',
    fontsize=11)
axs[2, 1].set_xlabel('Δ')
axs[2, 1].set_ylabel('Posterior Probability Density')

axs[2, 2].axis('off')  # Empty final cell

# Optional: Global caption
# fig.text(
#     0.5,
#     0.01,
#     "Each panel shows the posterior distribution of the probability (θ) that a participant reacquires a procedural strategy.\n"
#     "Δ represents the difference between experiments or conditions. Red lines = 95% credible interval; dashed line = null.",
#     ha='center',
#     fontsize=10)

# Print summaries
print(f"Exp 1 — Relearn − New Learn: 95% CI = {ci_exp1[0]:.3f} to {ci_exp1[1]:.3f}, P(Δ > 0) = {prob_exp1:.3f}")
print(f"Exp 2 — Relearn − New Learn: 95% CI = {ci_exp2[0]:.3f} to {ci_exp2[1]:.3f}, P(Δ > 0) = {prob_exp2:.3f}")

plt.savefig("../figures/bayesian_comparison.png", dpi=300)
plt.close()

