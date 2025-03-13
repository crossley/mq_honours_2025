import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
import patsy
from util_func_dbm import *


def load_data():

    dir_data = "../data"

    d_rec = []
    for f in os.listdir(dir_data):
        if f.endswith(".csv"):
            d = pd.read_csv(os.path.join(dir_data, f))
            d_rec.append(d)

    d = pd.concat(d_rec)

    d["cat"] = d["cat"].astype("category")
    d["sub_task"] = d["sub_task"].astype("category")

    d["effector"] = "NA"

    d["Effector"] = d["cat"].cat.rename_categories({
        107: "L1",
        97: "R1",
        108: "L2",
        115: "R2"
    })

    d["cat"] = d["cat"].astype(int)

    d.loc[d["cat"] == 107, "cat"] = 0
    d.loc[d["cat"] == 115, "cat"] = 1
    d.loc[d["cat"] == 97, "cat"] = 0
    d.loc[d["cat"] == 108, "cat"] = 1
    d.loc[d["resp"] == 107, "resp"] = 0
    d.loc[d["resp"] == 115, "resp"] = 1
    d.loc[d["resp"] == 97, "resp"] = 0
    d.loc[d["resp"] == 108, "resp"] = 1

    block_size = 100
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size

    d["acc"] = d["cat"] == d["resp"]

    d = d.sort_values(["condition", "subject", "block", "trial"])

    print(d.groupby(["condition"])["subject"].unique())
    print(d.groupby(["condition"])["subject"].nunique())

    return d


def make_fig_cats():

    d = load_data()

    d["cat"] = d["cat"].astype("category")
    d["sub_task"] = d["sub_task"].astype("category")

    # recode cat level names
    d["cat"] = d["cat"].cat.rename_categories({
        107: "L1",
        97: "R1",
        108: "L2",
        115: "R2"
    })

    d["condition"] = d["condition"].astype("category")
    d["condition"] = d["condition"].cat.rename_categories(
        ["Congruent", "Incongruent"])

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 10))
    sns.scatterplot(data=d[(d["condition"] == "Congruent")
                           & (d["sub_task"] == 1)],
                    x="x",
                    y="y",
                    hue="Effector",
                    style="Effector",
                    ax=ax[0, 0])
    sns.scatterplot(data=d[(d["condition"] == "Congruent")
                           & (d["sub_task"] == 2)],
                    x="x",
                    y="y",
                    hue="Effector",
                    style="Effector",
                    ax=ax[0, 1])
    sns.scatterplot(data=d[(d["condition"] == "Incongruent")
                           & (d["sub_task"] == 1)],
                    x="x",
                    y="y",
                    hue="Effector",
                    style="Effector",
                    ax=ax[1, 0])
    sns.scatterplot(data=d[(d["condition"] == "Incongruent")
                           & (d["sub_task"] == 2)],
                    x="x",
                    y="y",
                    hue="Effector",
                    style="Effector",
                    ax=ax[1, 1])
    sns.move_legend(ax[0, 0], "upper left")
    sns.move_legend(ax[0, 1], "upper left")
    sns.move_legend(ax[1, 0], "upper left")
    sns.move_legend(ax[0, 1], "upper left")
    ax[0, 0].set_title("Congruent subtask 1")
    ax[0, 1].set_title("Congruent subtask 2")
    ax[1, 0].set_title("Incongruent subtask 1")
    ax[1, 1].set_title("Incongruent subtask 2")
    [x.set_xticks([]) for x in ax.flatten()]
    [x.set_yticks([]) for x in ax.flatten()]
    [x.set_xlabel("") for x in ax.flatten()]
    [x.set_ylabel("") for x in ax.flatten()]
    for i, ax_ in enumerate(ax.flatten()):
        ax_.text(-0.1, 1.0, chr(65 + i), transform=ax_.transAxes, size=20)
    plt.savefig("../figures/fig_categories_stim_space.png")
    plt.close()


def fit_dbm_models(d):

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

    # NOTE: only look at final block
    dbm = (d[d["block"] == 3].groupby([
        "condition", "subject", "sub_task", "block"
    ]).apply(fit_dbm, models, side, k, n, model_names).reset_index())

    dbm.to_csv("../dbm_fits/dbm_results.csv")


def get_best_model_class(dbm):

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        best_model = np.unique(model[bic == bic.min()])[0]
        x["best_model"] = best_model
        return x

    dbm = dbm.groupby(["condition", "subject", "sub_task",
                       "block"]).apply(assign_best_model)

    dd = dbm.loc[dbm["model"] == dbm["best_model"]]

    ddd = dd[["condition", "subject", "sub_task", "block",
              "best_model"]].drop_duplicates()

    ddd["best_model_class"] = ddd["best_model"].str.split("_").str[1]

    ddd.loc[ddd["best_model_class"] != "glc",
            "best_model_class"] = "rule-based"

    ddd.loc[ddd["best_model_class"] == "glc",
            "best_model_class"] = "procedural"

    ddd = ddd.reset_index(drop=True)

    def get_best_model_class_2(x):
        if np.isin("rule-based", x["best_model_class"].to_numpy()):
            x["best_model_class_2"] = "rule-based"
        else:
            x["best_model_class_2"] = "procedural"

        return x

    ddd = ddd.groupby(["condition", "subject"
                       ]).apply(get_best_model_class_2).reset_index(drop=True)

    ddd["best_model_class"] = ddd["best_model_class"].astype("category")
    ddd["sub_task"] = ddd["sub_task"].astype("category")
    ddd["best_model_class_2"] = ddd["best_model_class_2"].astype("category")

    return dd, ddd


def make_fig_dbm(d, dd, ddd):

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
    ax[0, 2].set_title("Congruent")
    ax[1, 2].set_title("Incongruent")
    ax[0, 2].set_ylabel("Proportion of participants")
    ax[1, 2].set_ylabel("Proportion of participants")
    ax[0, 2].set_xlabel("Model class")
    ax[1, 2].set_xlabel("Model class")

    # plot bounds
    for c in dd["condition"].unique():
        dc = dd[dd["condition"] == c]
        for s in dc["subject"].unique():
            ds = dc[dc["subject"] == s]
            for st in ds["sub_task"].unique():

                if c == "4F4K_congruent":
                    plot_title = f"Congruent subtask {st}"
                    if st == 1:
                        ax_ = ax[0, 0]
                    else:
                        ax_ = ax[0, 1]
                else:
                    plot_title = f"Incongruent subtask {st}"
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

    # turn off axis labels and ticks for ax[:, 0:2]
    [x.set_xticks([]) for x in ax[:, 0:2].flatten()]
    [x.set_yticks([]) for x in ax[:, 0:2].flatten()]
    [x.set_xlabel("") for x in ax[:, 0:2].flatten()]
    [x.set_ylabel("") for x in ax[:, 0:2].flatten()]

    for i, ax_ in enumerate(ax.flatten()):
        ax_.text(-0.1, 1.05, chr(65 + i), transform=ax_.transAxes, size=20)

    plt.tight_layout()
    plt.savefig("../figures/fig_dbm.png")
    plt.close()

    results = pg.chi2_independence(ddd, x='condition', y='best_model_class_2')
    print(results)



def make_fig_accuracy_per_block_by_model(d, dd, ddd):

    # NOTE: make learning curve for procedural and rule-based models
    block_size = 25
    d["block"] = d.groupby(["condition", "subject"
                            ]).cumcount() // block_size + 1

    d = d.merge(ddd[["condition", "subject", "sub_task",
                     "best_model_class_2"]],
                on=["condition", "subject", "sub_task"])

    dd = d.groupby(
        ["best_model_class_2", "condition", "subject", "sub_task", "block"],
        observed=True)["acc"].mean().reset_index()

    dd["condition"] = dd["condition"].astype("category")
    dd["condition"] = dd["condition"].cat.rename_categories(
        ["Congruent", "Incongruent"])

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))

    sns.lineplot(data=dd,
                 x="block",
                 y="acc",
                 style="sub_task",
                 hue="condition",
                 errorbar=None,
                 legend=False,
                 alpha=0.25,
                 ax=ax[0, 0])
    sns.lineplot(data=dd,
                 x="block",
                 y="acc",
                 hue="condition",
                 errorbar="se",
                 err_style="bars",
                 ax=ax[0, 0])

    sns.lineplot(data=dd[dd["best_model_class_2"] == "rule-based"],
                 x="block",
                 y="acc",
                 style="sub_task",
                 hue="condition",
                 errorbar=None,
                 legend=False,
                 alpha=0.25,
                 ax=ax[0, 1])
    sns.lineplot(data=dd[dd["best_model_class_2"] == "rule-based"],
                 x="block",
                 y="acc",
                 hue="condition",
                 errorbar="se",
                 err_style="bars",
                 ax=ax[0, 1])

    sns.lineplot(data=dd[dd["best_model_class_2"] == "procedural"],
                 x="block",
                 y="acc",
                 style="sub_task",
                 hue="condition",
                 errorbar=None,
                 legend=False,
                 alpha=0.25,
                 ax=ax[0, 2])
    sns.lineplot(data=dd[dd["best_model_class_2"] == "procedural"],
                 x="block",
                 y="acc",
                 hue="condition",
                 errorbar="se",
                 err_style="bars",
                 ax=ax[0, 2])

    [x.set_ylim(0.3, 1) for x in ax.flatten()]
    [x.set_xticks(dd.block.unique()) for x in ax.flatten()]
    [x.set_xlabel("Block") for x in ax.flatten()]
    [x.set_ylabel("Proportion correct") for x in ax.flatten()]
    [sns.move_legend(x, "lower right", ncols=1) for x in ax.flatten()]
    ax[0, 0].set_title("All participants")
    ax[0, 1].set_title("Rule-based participants")
    ax[0, 2].set_title("Procedural participants")
    for i, ax_ in enumerate(ax.flatten()):
        ax_.text(-0.17, 1.05, chr(65 + i), transform=ax_.transAxes, size=20)
    plt.tight_layout()
    plt.savefig("../figures/fig_accuracy_per_block_by_model.png")
    plt.close()


def report_stats_learning_curve(d, dd, ddd):

    d = load_data()

    d = d.merge(ddd[["condition", "subject", "sub_task",
                     "best_model_class_2"]],
                on=["condition", "subject", "sub_task"])

    subs_procedural = d[d["best_model_class_2"] ==
                        "procedural"]["subject"].unique()

    subs_rule_based = d[d["best_model_class_2"] ==
                        "rule-based"]["subject"].unique()

    d["model_class"] = "NA"
    d.loc[d["subject"].isin(subs_procedural), "model_class"] = "procedural"
    d.loc[d["subject"].isin(subs_rule_based), "model_class"] = "rule-based"

    # d = d[d["model_class"] == "procedural"]

    d = d[['condition', 'subject', 'sub_task', 'trial', 'acc', 'model_class']]

    d = pd.get_dummies(d,
                       columns=["condition", "sub_task", "model_class"],
                       drop_first=True,
                       dtype=int)

    d = d.rename(columns={"condition_4F4K_incongruent": "condition"})
    d = d.rename(columns={"sub_task_2": "sub_task"})
    d = d.rename(columns={"model_class_rule-based": "model_class"})

    d["trial"] = d["trial"] + 1

    y = d["acc"]
    X = patsy.dmatrix("condition * model_class * np.log(trial)",
                      data=d,
                      return_type="dataframe")

    model = sm.Logit(y, X)
    res = model.fit()
    print(res.summary())
    print(res.summary().as_latex())

    d["acc_pred"] = res.predict(X)


def make_fig_switch_cost(d, dd, ddd):

    d = load_data()

    d = d.merge(ddd[["condition", "subject", "sub_task",
                     "best_model_class_2"]],
                on=["condition", "subject", "sub_task"])

    subs_procedural = d[d["best_model_class_2"] ==
                        "procedural"]["subject"].unique()

    subs_rule_based = d[d["best_model_class_2"] ==
                        "rule-based"]["subject"].unique()

    d["model_class"] = "NA"
    d.loc[d["subject"].isin(subs_procedural), "model_class"] = "procedural"
    d.loc[d["subject"].isin(subs_rule_based), "model_class"] = "rule-based"

    d = d[[
        'condition', 'subject', 'sub_task', 'trial', 'acc', 'rt', 'model_class'
    ]]

    d["stay"] = d["sub_task"].shift(1) == d["sub_task"]

    dd = d.groupby(["condition", "subject", "stay", "model_class"]).agg({
        "acc":
        "mean",
        "rt":
        "mean"
    }).reset_index()

    # compute accuracy switch cost per subject
    dd["switch_cost_acc"] = dd.groupby(["condition", "subject"])["acc"].diff()
    dd["switch_cost_rt"] = -dd.groupby(["condition", "subject"])["rt"].diff()

    # select only rows where switch cost is not NaN
    dd = dd[~np.isnan(dd["switch_cost_acc"])][[
        "condition", "subject", "model_class", "switch_cost_acc",
        "switch_cost_rt"
    ]]

    dd["condition"] = dd["condition"].astype("category")
    dd["condition"] = dd["condition"].cat.rename_categories(
        ["Congruent", "Incongruent"])

    tmp = dd.copy()
    tmp["model_class"] = "all"
    dd = pd.concat([tmp, dd])

    dd["model_class"] = dd["model_class"].astype("category")

    dd["model_class"] = dd["model_class"].cat.rename_categories([
        "All participants", "Rule-based participants",
        "Procedural participants"
    ])

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.4, bottom=0.15)
    sns.pointplot(data=dd,
                  x="model_class",
                  y="switch_cost_acc",
                  hue="condition",
                  linestyles="none",
                  dodge=0.15,
                  legend=True,
                  ax=ax[0, 0])
    sns.pointplot(data=dd,
                  x="model_class",
                  y="switch_cost_rt",
                  hue="condition",
                  linestyles="none",
                  dodge=0.15,
                  legend=True,
                  ax=ax[0, 1])
    sns.move_legend(ax[0, 0], "upper left", ncol=2)
    sns.move_legend(ax[0, 1], "upper left", ncol=2)
    ax[0, 0].set_ylabel("Accuracy switch cost")
    ax[0, 1].set_ylabel("Response time switch cost")
    [x.set_xlabel("") for x in ax.flatten()]
    [x.tick_params(axis='x', rotation=15) for x in ax.flatten()]
    for i, ax_ in enumerate(ax.flatten()):
        ax_.text(-0.1, 1.05, chr(65 + i), transform=ax_.transAxes, size=20)
    plt.savefig("../figures/fig_switch_cost.png")
    plt.close()

    dd = dd[dd["model_class"] != "All participants"]
    dd["model_class"] = dd["model_class"].cat.remove_unused_categories()

    res = pg.anova(data=dd,
                   dv="switch_cost_acc",
                   between=["condition", "model_class"],
                   ss_type=3).round(2)
    print(res)

    res = pg.anova(data=dd,
                   dv="switch_cost_rt",
                   between=["condition", "model_class"],
                   ss_type=3).round(2)
    print(res)
