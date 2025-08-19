import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import CubicSpline
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
from patsy.contrasts import Diff, Treatment


def interpolate_movements(d):
    t = d["t"]
    x = d["x"]
    y = d["y"]
    v = d["v"]

    xs = CubicSpline(t, x)
    ys = CubicSpline(t, y)
    vs = CubicSpline(t, v)

    tt = np.linspace(t.min(), t.max(), 100)
    xx = xs(tt)
    yy = ys(tt)
    vv = vs(tt)

    relsamp = np.arange(0, tt.shape[0], 1)

    dd = pd.DataFrame({"relsamp": relsamp, "t": tt, "x": xx, "y": yy, "v": vv})
    dd["condition"] = d["condition"].unique()[0]
    dd["subject"] = d["subject"].unique()[0]
    dd["trial"] = d["trial"].unique()[0]
    dd["phase"] = d["phase"].unique()[0]
    dd["su"] = d["su"].unique()[0]
    dd["imv"] = d["imv"].unique()[0]
    dd["emv"] = d["emv"].unique()[0]

    return dd


def compute_kinematics(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()

    x = x - x[0]
    y = y - y[0]
    y = -y

    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x)) * 180 / np.pi

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)

    v_peak = v.max()
    # ts = t[v > (0.05 * v_peak)][0]
    ts = t[r > 0.1 * r.max()][0]

    imv = theta[(t >= ts) & (t <= ts + 0.1)].mean()
    emv = theta[-1]

    d["x"] = x
    d["y"] = y
    d["v"] = v
    d["imv"] = 90 - imv
    d["emv"] = 90 - emv

    return d


dir_data = "../data/"

d_rec = []

for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25,
          26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:

    f_trl = "sub_{}_data.csv".format(s)
    f_mv = "sub_{}_data_move.csv".format(s)

    d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
    d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

    d_trl = d_trl.sort_values(["condition", "subject", "trial"])
    d_mv = d_mv.sort_values(["condition", "subject", "t", "trial"])

    d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
    x_start = d_hold.x.mean()
    y_start = d_hold.y.mean()

    d_mv = d_mv[d_mv["state"].isin(["state_moving"])]

    phase = np.zeros(d_trl["trial"].nunique())
    phase[:30] = 1
    phase[30:130] = 2
    phase[130:180] = 3
    phase[180:230] = 4
    phase[230:330] = 5
    phase[330:380] = 6
    phase[380:] = 7
    d_trl["phase"] = phase

    d_trl["su"] = d_trl["su"].astype("category")
    d_trl["ep"] = (d_trl["ep"] * 180 / np.pi) + 90
    d_trl["rotation"] = d_trl["rotation"] * 180 / np.pi

    d = pd.merge(d_mv,
                 d_trl,
                 how="outer",
                 on=["condition", "subject", "trial"])

    d = d.groupby(["condition", "subject", "trial"],
                  group_keys=False).apply(compute_kinematics)

    d_rec.append(d)

d = pd.concat(d_rec)

d.groupby(["condition"])["subject"].unique()
d.groupby(["condition"])["subject"].nunique()

d.sort_values(["condition", "subject", "trial", "t"], inplace=True)

# for s in d["subject"].unique():
#     ds = d[d["subject"] == s]
#     fig, ax = plt.subplots(3, 1, squeeze=False)
#     sns.scatterplot(data=ds, x="trial", y="rotation", hue="condition", ax=ax[0, 0])
#     sns.scatterplot(data=ds, x="trial", y="su", hue="condition", ax=ax[1, 0])
#     sns.scatterplot(data=ds, x="trial", y="imv", hue="condition", ax=ax[2, 0])
#     ax[1, 0].invert_yaxis()
#     plt.suptitle("Subject {}".format(s))
#     plt.show()

d.loc[(d["condition"] == "blocked")
      & np.isin(d["subject"], [1, 5, 9, 13, 15, 19, 21, 25, 29, 33, 35, 39]),
      "condition"] = "Blocked - Low High"

d.loc[(d["condition"] == "blocked")
      & np.isin(d["subject"], [3, 7, 11, 17, 23, 27, 31, 37]),
      "condition"] = "Blocked - High Low"

d.groupby(["condition"])["subject"].unique()
d.groupby(["condition"])["subject"].nunique()
d.groupby(["condition", "subject"])["trial"].nunique()

# NOTE: create by trial frame
dp = d[["condition", "subject", "trial", "phase", "su", "emv", "rotation"]].drop_duplicates()


def identify_outliers(x):
    x["outlier"] = False
    # nsd = 2.5
    # x.loc[(np.abs(x["emv"]) - x["emv"].mean()) > nsd * np.std(x["emv"]), "outlier"] = True
    x.loc[np.abs(x["emv"]) > 70, "outlier"] = True
    return x


dp = dp.groupby(["condition", "subject"]).apply(identify_outliers).reset_index(drop=True)
dp.groupby(["condition", "subject"])["outlier"].sum()
dp = dp[dp["outlier"] == False]
dp = dp.sort_values(["condition", "subject", "trial"])


def add_prev(x):
    x["su_prev"] = x["su"].shift(1)
    x["delta_emv"] = np.diff(x["emv"].to_numpy(), prepend=0)
    x["movement_error"] = -x["rotation"] + x["emv"]
    x["movement_error_prev"] = x["movement_error"].shift(1)
    return x


dp = dp.groupby(["condition", "subject"], group_keys=False).apply(add_prev)

# NOTE: inspect individual subjects --- measures
for i, s in enumerate(dp["subject"].unique()):

    ds = dp[dp["subject"] == s].copy()

    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(5, 12))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    sns.scatterplot(
        data=ds,
        x="trial",
        y="emv",
        hue="su_prev",
        markers=True,
        legend="full",
        ax=ax[0, 0],
    )
    sns.scatterplot(
        data=ds,
        x="trial",
        y="movement_error",
        hue="su_prev",
        markers=True,
        legend=False,
        ax=ax[1, 0],
    )
    sns.scatterplot(
        data=ds,
        x="trial",
        y="delta_emv",
        hue="su_prev",
        markers=True,
        legend=False,
        ax=ax[2, 0],
    )
    [
        sns.lineplot(
            data=ds,
            x="trial",
            y="rotation",
            hue="condition",
            palette=['k'],
            legend=False,
            ax=ax_,
        ) for ax_ in [ax[0, 0], ax[1, 0], ax[2, 0]]
    ]

    ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)

    plt.savefig("../figures/fig_measures_sub_" + str(s) + ".png")
    plt.close()

# NOTE: inspect individual subjects --- scatter
for i, s in enumerate(dp["subject"].unique()):

    ds = dp[dp["subject"] == s].copy()

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    sns.scatterplot(
        data=ds,
        x="trial",
        y="emv",
        hue="su_prev",
        markers=True,
        legend="full",
        ax=ax[0, 0],
    )
    sns.lineplot(data=ds,
                 x="trial",
                 y="rotation",
                 hue="condition",
                 palette=['k'],
                 legend=False,
                 ax=ax[0, 0])

    for j, ph in enumerate([2, 5]):
        ds = ds[~ds["su_prev"].isna()]
        su_levels = np.sort(ds["su_prev"].unique())
        dss = ds[ds["phase"] == ph].copy()
        for k, su in enumerate(dss["su_prev"].unique()):
            dsss = dss[dss["su_prev"] == su].copy()
            sns.scatterplot(
                data=dsss,
                x="movement_error_prev",
                y="delta_emv",
                hue="su_prev",
                legend="full",
                ax=ax[0, j + 1],
            )
            sns.regplot(
                data=dsss,
                x="movement_error_prev",
                y="delta_emv",
                scatter=False,
                color=sns.color_palette()[np.where(su_levels == su)[0][0]],
                ax=ax[0, j + 1],
            )

    ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
    ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
    ax[0, 2].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)

    plt.savefig("../figures/fig_scatter_sub_" + str(s) + ".png")
    plt.close()

# NOTE: Exclude ppts that have abberant movements
# subs_exc = [...]
# dp = dp[~np.isin(dp["subject"], subs_exc)]

# NOTE: average over subjects
dpp = dp.groupby(["condition", "trial", "phase", "su_prev"], observed=True)[[
    "emv", "delta_emv", "movement_error", "movement_error_prev", "rotation"
]].mean().reset_index()

# dp.to_csv("../data_summary/summary_per_trial_per_subject.csv")
# dpp.to_csv("../data_summary/summary_per_trial.csv")

fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 12))
ax = ax.T
fig.subplots_adjust(wspace=0.3, hspace=0.5)
sns.scatterplot(
    data=dpp[dpp["condition"] == "Blocked - High Low"],
    x="trial",
    y="emv",
    style="phase",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dpp[dpp["condition"] == "Blocked - Low High"],
    x="trial",
    y="emv",
    style="phase",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 1],
)
sns.scatterplot(
    data=dpp[dpp["condition"] == "interleaved"],
    x="trial",
    y="emv",
    style="phase",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 2],
)
[x.set_ylim(-10, 40) for x in [ax[0, 0], ax[0, 1], ax[0, 2]]]
[x.set_xlabel("Trial") for x in [ax[0, 0], ax[0, 1], ax[0, 2]]]
[x.set_ylabel("Endppoint Movement Vector") for x in [ax[0, 0], ax[0, 1], ax[0, 2]]]
[
    sns.lineplot(
        data=dpp[dpp["condition"] != "interleaved"],
        x="trial",
        y="rotation",
        hue="condition",
        palette=['k'],
        legend=False,
        ax=ax_,
    ) for ax_ in [ax[0, 0], ax[0, 1], ax[0, 2]]
]
ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
ax[0, 2].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
plt.savefig("../figures/summary_per_trial.png")
plt.close()

# NOTE: scatter slope analysis

# adapt 1 is trials 30:130
ph = 2
dppp1 = dp[(dp["condition"] == "Blocked - Low High") & (dp["phase"] == ph) & (dp["trial"] < 35)].copy()
dppp2 = dp[(dp["condition"] == "Blocked - High Low") & (dp["phase"] == ph) & (dp["trial"] < 35)].copy()
dppp3 = dp[(dp["condition"] == "interleaved") & (dp["phase"] == ph)        & (dp["trial"] < 35)].copy()
fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
fig.subplots_adjust(wspace=0.3, hspace=0.5)
sns.scatterplot(
    data=dppp1,
    x="movement_error_prev",
    y="delta_emv",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dppp2,
    x="movement_error_prev",
    y="delta_emv",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 1],
)
sns.scatterplot(
    data=dppp3,
    x="movement_error_prev",
    y="delta_emv",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 2],
)
ax[0, 0].set_title("Blocked - Low High")
ax[0, 1].set_title("Blocked - High Low")
ax[0, 2].set_title("Interleaved")
[x.set_xlim(-20, 5) for x in ax.flatten()]
[x.set_ylim(-20, 20) for x in ax.flatten()]
[x.set_xlabel("Movement error previous trial") for x in ax.flatten()]
[x.set_ylabel("Change in EMV") for x in ax.flatten()]
plt.savefig("../figures/summary_scatter_slope.png")
plt.close()

# NOTE: statsmodels
mod_formula = "delta_emv ~ "
mod_formula += "C(su_prev, Diff) * movement_error_prev + "
mod_formula += "np.log(trial) + "
mod_formula += "1"

ph = 2

# NOTE: set condition here
dppp = dpp[(dpp["condition"] == "Blocked - Low High") & (dpp["phase"] == ph) & (dpp["trial"] < 60)].copy()
# dppp = dpp[(dpp["condition"] == "Blocked - High Low") & (dpp["phase"] == ph) & (dpp["trial"] < 60)].copy()
# dppp = dpp[(dpp["condition"] == "interleaved") & (dpp["phase"] == ph) & (dpp["trial"] < 60)].copy()

mod = smf.ols(mod_formula, data=dppp)
res_sm = mod.fit()
print(res_sm.summary())

dppp["delta_emv_pred"] = res_sm.model.predict(res_sm.params, res_sm.model.exog)

# plot obs and pred overliad
fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(6, 6))
fig.subplots_adjust(wspace=0.3, hspace=0.5)
sns.scatterplot(data=dppp,
                x="trial",
                y="delta_emv",
                hue="su_prev",
                markers=True,
                ax=ax[0, 0])
sns.scatterplot(data=dppp,
                x="trial",
                y="delta_emv_pred",
                hue="su_prev",
                markers=True,
                ax=ax[0, 1])
plt.show()
