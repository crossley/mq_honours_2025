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
            d["block"] = np.floor(d["trial"] / 25).astype(int)
            d["acc"] = d["cat"] == d["resp"]
            d["phase"] = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"
                                                                     ] * 299
            d_rec.append(d)

    d = pd.concat(d_rec, ignore_index=True)

    # NOTE: Fix bug in code for first 18 ppts
    d.loc[(d["condition"] == "new_learn") & (d["subject"] <= 18),
          "experiment"] = 2

    d.loc[d["cat"] == "A", "cat"] = 0
    d.loc[d["cat"] == "B", "cat"] = 1
    d.loc[d["resp"] == "A", "resp"] = 0
    d.loc[d["resp"] == "B", "resp"] = 1
    d["cat"] = d["cat"].astype(int)
    d["resp"] = d["resp"].astype(int)

    block_size = 100
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size

    d = d.sort_values(["condition", "subject", "block", "trial"])

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
        dbm = (d.groupby(["condition", "subject",
                          "block"]).apply(fit_dbm, models, side, k, n,
                                          model_names).reset_index())

        dbm.to_csv("../dbm_fits/dbm_results.csv")

    else:
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")

    dbm = dbm.groupby(["condition", "subject",
                       "block"]).apply(assign_best_model)

    dd = dbm.loc[dbm["model"] == dbm["best_model"]]

    ddd = dd[["condition", "subject", "block",
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

    ddd = ddd.groupby(["condition", "subject"
                       ]).apply(get_best_model_class_2).reset_index(drop=True)
    ddd["best_model_class_2"] = ddd["best_model_class_2"].astype("category")

    dcat = d[["condition", "sub_task", "x", "y", "cat"]].drop_duplicates()
    dcat["effector"] = "None"
    dcat.loc[(dcat["condition"] == "4F4K_congruent") & (dcat["sub_task"] == 1)
             & (dcat["cat"] == 0), "effector"] = "L1"
    dcat.loc[(dcat["condition"] == "4F4K_congruent") & (dcat["sub_task"] == 1)
             & (dcat["cat"] == 1), "effector"] = "R1"
    dcat.loc[(dcat["condition"] == "4F4K_congruent") & (dcat["sub_task"] == 2)
             & (dcat["cat"] == 0), "effector"] = "R2"
    dcat.loc[(dcat["condition"] == "4F4K_congruent") & (dcat["sub_task"] == 2)
             & (dcat["cat"] == 1), "effector"] = "L2"
    dcat.loc[(dcat["condition"] == "4F4K_incongruent") &
             (dcat["sub_task"] == 1) & (dcat["cat"] == 0), "effector"] = "L1"
    dcat.loc[(dcat["condition"] == "4F4K_incongruent") &
             (dcat["sub_task"] == 1) & (dcat["cat"] == 1), "effector"] = "R1"
    dcat.loc[(dcat["condition"] == "4F4K_incongruent") &
             (dcat["sub_task"] == 2) & (dcat["cat"] == 0), "effector"] = "R2"
    dcat.loc[(dcat["condition"] == "4F4K_incongruent") &
             (dcat["sub_task"] == 2) & (dcat["cat"] == 1), "effector"] = "L2"
    dcat["effector"] = dcat["effector"].astype("category")
    dcat["effector"] = dcat["effector"].cat.reorder_categories(
        ["L1", "R1", "L2", "R2"])

    fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(12, 8))
    # plot categories
    sns.scatterplot(data=dcat[(dcat["condition"] == "4F4K_congruent")
                              & (dcat["sub_task"] == 1)],
                    x="x",
                    y="y",
                    hue="effector",
                    legend=True,
                    ax=ax[0, 0])
    sns.scatterplot(data=dcat[(dcat["condition"] == "4F4K_congruent")
                              & (dcat["sub_task"] == 2)],
                    x="x",
                    y="y",
                    hue="effector",
                    legend=True,
                    ax=ax[0, 1])
    sns.scatterplot(data=dcat[(dcat["condition"] == "4F4K_incongruent")
                              & (dcat["sub_task"] == 1)],
                    x="x",
                    y="y",
                    hue="effector",
                    legend=True,
                    ax=ax[1, 0])
    sns.scatterplot(data=dcat[(dcat["condition"] == "4F4K_incongruent")
                              & (dcat["sub_task"] == 2)],
                    x="x",
                    y="y",
                    hue="effector",
                    legend=True,
                    ax=ax[1, 1])

    # plot counts
    sns.countplot(data=ddd[ddd["condition"] == "4F4K_congruent"],
                  x='best_model_class_2',
                  stat="proportion",
                  ax=ax[0, 2])
    sns.countplot(data=ddd[ddd["condition"] == "4F4K_incongruent"],
                  x='best_model_class_2',
                  stat="proportion",
                  ax=ax[1, 2])
    ax[0, 2].set - title("4F4K_congruent")
    ax[1, 2].set - title("4F4K_incongruent")

    # plot bounds
    for c in dd["condition"].unique():
        dc = dd[dd["condition"] == c]
        for s in dc["subject"].unique():
            ds = dc[dc["subject"] == s]
            for st in ds["sub_task"].unique():

                plot_title = f"Condition: {c} - Sub-Task: {st}"

                if c == "4F4K_congruent":
                    if st == 1:
                        ax_ = ax[0, 0]
                    else:
                        ax_ = ax[0, 1]
                else:
                    if st == 1:
                        ax_ = ax[1, 0]
                    else:
                        ax_ = ax[1, 1]

                x = dd.loc[(dd["condition"] == c) & (dd["subject"] == s) &
                           (dd["sub_task"] == st) & (dd["block"] == 3)]

                best_model = x["best_model"].to_numpy()[0]

                if best_model in ("nll_unix_0", "nll_unix_1"):
                    xc = x["p"].to_numpy()[0]
                    ax_.plot([xc, xc], [0, 100], "--k")

                elif best_model in ("nll_uniy_0", "nll_uniy_1"):
                    yc = x["p"].to_numpy()[0]
                    ax_.plot([0, 100], [yc, yc], "--k")

                elif best_model in ("nll_glc_0", "nll_glc_1"):
                    # a1 * x + a2 * y + b = 0
                    # y = -(a1 * x + b) / a2
                    a1 = x["p"].to_numpy()[0]
                    a2 = np.sqrt(1 - a1**2)
                    b = x["p"].to_numpy()[1]
                    ax_.plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], "-k")

                elif best_model in ("nll_gcc_eq_0", "nll_gcc_eq_1",
                                    "nll_gcc_eq_2", "nll_gcc_eq_3"):
                    xc = x["p"].to_numpy()[0]
                    yc = x["p"].to_numpy()[1]
                    ax_.plot([xc, xc], [0, 100], "-k")
                    ax_.plot([0, 100], [yc, yc], "-k")

                ax_.set_xlim(-5, 105)
                ax_.set_ylim(-5, 105)
                ax_.set_title(plot_title)

    plt.tight_layout()
    plt.show()
    # plt.savefig("../figures/fig_dbm.pdf")
