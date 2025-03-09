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

for s in range(13, 40):

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
#     fig, ax = plt.subplots(1, 1, squeeze=False)
#     ax[0, 0].plot(d[d["subject"] == s]["su"])
#     ax[0, 0].set_title(f"Subject {s}")
#     plt.show()

# high low
# 13, 15, 17, 25, 31, 35

# low high
# 19, 21, 23, 27, 29, 33, 37, 39

d.loc[(d["condition"] == "blocked")
      & np.isin(d["subject"], [13, 15, 17, 25, 31, 35]),
      "condition"] = "Blocked - High low"
d.loc[(d["condition"] == "blocked")
      & np.isin(d["subject"], [19, 21, 23, 27, 29, 33, 37, 39]),
      "condition"] = "Blocked - Low High"

d.groupby(["condition"])["subject"].unique()

# NOTE: because of bug in experiment (15 deg rotation
# applied twice)
d["rotation"] = d["rotation"] * 2

d.groupby(["condition", "subject"])["trial"].nunique()

# NOTE: create by trial frame
dp = d[["condition", "subject", "trial", "phase", "su", "emv",
        "rotation"]].drop_duplicates()


def identify_outliers(x):
    x["outlier"] = False
    # nsd = 2.5
    # x.loc[(np.abs(x["emv"]) - x["emv"].mean()) > nsd * np.std(x["emv"]), "outlier"] = True
    x.loc[np.abs(x["emv"]) > 70, "outlier"] = True
    return x


dp = dp.groupby(["condition",
                 "subject"]).apply(identify_outliers).reset_index(drop=True)

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
subs_exc = [2, 5, 20]
dp = dp[~np.isin(dp["subject"], subs_exc)]

# NOTE: average over subjects
dpp = dp.groupby(["condition", "trial", "phase", "su_prev"], observed=True)[[
    "emv", "delta_emv", "movement_error", "movement_error_prev", "rotation"
]].mean().reset_index()

dp.to_csv("../data_summary/summary_per_trial_per_subject.csv")
dpp.to_csv("../data_summary/summary_per_trial.csv")

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 4))
fig.subplots_adjust(wspace=0.3, hspace=0.5)
sns.scatterplot(
    data=dpp[dpp["condition"] != "interleaved"],
    x="trial",
    y="emv",
    style="condition",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dpp[dpp["condition"] == "interleaved"],
    x="trial",
    y="emv",
    style="condition",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 1],
)
[x.set_ylim(-10, 50) for x in [ax[0, 0], ax[0, 1]]]
[x.set_xlabel("Trial") for x in [ax[0, 0], ax[0, 1]]]
[x.set_ylabel("Endppoint Movement Vector") for x in [ax[0, 0], ax[0, 1]]]
[
    sns.lineplot(
        data=dpp[dpp["condition"] != "interleaved"],
        x="trial",
        y="rotation",
        hue="condition",
        palette=['k'],
        legend=False,
        ax=ax_,
    ) for ax_ in [ax[0, 0], ax[0, 1]]
]
ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
plt.savefig("../figures/summary_per_trial.png")
plt.close()

# NOTE: visually identify rule users
subs_rb = [19, 23, 25, 29, 33]
dp["rb"] = False
dp.loc[np.isin(dp["subject"], subs_rb), "rb"] = True

dpp = dp.groupby(["condition", "trial", "phase", "su_prev", "rb"],
                 observed=True)[[
                     "emv", "delta_emv", "movement_error",
                     "movement_error_prev", "rotation"
                 ]].mean().reset_index()

dppp = dpp[dpp["condition"] != "interleaved"].copy()

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 4))
fig.subplots_adjust(wspace=0.3, hspace=0.5)
sns.scatterplot(
    data=dppp[dppp["rb"] == False],
    x="trial",
    y="emv",
    style="condition",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 0],
)
sns.scatterplot(
    data=dppp[dppp["rb"] == True],
    x="trial",
    y="emv",
    style="condition",
    hue="su_prev",
    markers=True,
    legend="full",
    ax=ax[0, 1],
)
[x.set_ylim(-10, 50) for x in [ax[0, 0], ax[0, 1]]]
[x.set_xlabel("Trial") for x in [ax[0, 0], ax[0, 1]]]
[x.set_ylabel("Endppoint Movement Vector") for x in [ax[0, 0], ax[0, 1]]]
[
    sns.lineplot(
        data=dppp[dppp["condition"] != "interleaved"],
        x="trial",
        y="rotation",
        hue="condition",
        palette=['k'],
        legend=False,
        ax=ax_,
    ) for ax_ in [ax[0, 0], ax[0, 1]]
]
ax[0, 0].set_title("Rule non-users")
ax[0, 1].set_title("Rule users")
ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
plt.savefig("../figures/summary_per_trial_rb.png")
plt.close()

# NOTE: scatter slope analysis
# adapt 1 is trials 30:130
dppp1 = dp[(dp["condition"] != "interleaved") & (dp["phase"] == ph) &
          (dp["rb"] == False) & (dp["trial"] < 50)].copy()
dppp2 = dp[(dp["condition"] != "interleaved") & (dp["phase"] == ph) &
           (dp["rb"] == True) & (dp["trial"] < 50)].copy()
dppp3 = dp[(dp["condition"] == "interleaved") & (dp["phase"] == ph) &
           (dp["rb"] == False) & (dp["trial"] < 50)].copy()
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
ax[0, 0].set_title("Blocked - non-rule users")
ax[0, 1].set_title("Blocked - rule users")
ax[0, 2].set_title("Interleaved")
[x.set_xlabel("Movement error previous trial") for x in ax.flatten()]
[x.set_ylabel("Change in EMV") for x in ax.flatten()]
plt.savefig("../figures/summary_scatter_slope.png")
plt.close()

# NOTE: statsmodels
mod_formula = "emv ~ "
mod_formula += "C(su_prev, Diff) * movement_error_prev + "
mod_formula += "np.log(trial) + "
mod_formula += "1"

ph = 2

# NOTE: blocked
# dppp = dpp[(dpp["condition"] != "interleaved") & (dpp["phase"] == ph)].copy()
dppp = dpp[(dpp["condition"] != "interleaved") & (dpp["phase"] == ph) &
           (dpp["rb"] == False) & (dpp["trial"] < 60)].copy()

mod = smf.ols(mod_formula, data=dppp)
res_sm = mod.fit()
print(res_sm.summary())

dppp["emv_pred"] = res_sm.model.predict(res_sm.params, res_sm.model.exog)

# plot obs and pred overliad
fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(6, 6))
fig.subplots_adjust(wspace=0.3, hspace=0.5)
sns.scatterplot(data=dppp,
                x="trial",
                y="emv",
                hue="su_prev",
                markers=True,
                ax=ax[0, 0])
sns.scatterplot(data=dppp,
                x="trial",
                y="emv_pred",
                hue="su_prev",
                markers=True,
                ax=ax[0, 1])
plt.show()

# NOTE: interleaved
dppp = dpp[(dpp["condition"] == "interleaved") & (dpp["phase"] == ph)].copy()

mod = smf.ols(mod_formula, data=dppp)
res_sm = mod.fit()
print(res_sm.summary())

dppp["emv_pred"] = res_sm.model.predict(res_sm.params, res_sm.model.exog)

# NOTE: 3-way anova approach
d = dp.copy()

d["phase_2"] = "None"
d.loc[np.isin(d["trial"], [31, 32, 33]), "phase_2"] = "rot_1"
d.loc[np.isin(d["trial"], [231, 232, 233]), "phase_2"] = "rot_2"
# d.loc[np.isin(d["trial"], [128, 129, 130]), "phase_2"] = "rot_1"
# d.loc[np.isin(d["trial"], [328, 329, 330]), "phase_2"] = "rot_2"

d.loc[d["condition"].str.contains("Blocked"), "condition"] = "blocked"

d = d[["condition", "subject", "phase_2", "su_prev", "emv",
       "delta_emv"]].copy()
d = d[d["phase_2"] != "None"]

dd = d.groupby(["condition", "subject", "phase_2", "su_prev"],
               observed=True)[["emv", "delta_emv"]].mean().reset_index()

dd["su_prev"] = dd["su_prev"].astype("category")
dd["condition"] = dd["condition"].astype("category")
dd["phase_2"] = dd["phase_2"].astype("category")

dd.to_csv("../data_summary/d_for_anova.csv")

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 4))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
sns.pointplot(data=dd[dd["condition"] == "blocked"],
              x="phase_2",
              y="emv",
              hue="su_prev",
              errorbar="se",
              legend="full",
              ax=ax[0, 0])
sns.pointplot(data=dd[dd["condition"] == "interleaved"],
              x="phase_2",
              y="emv",
              hue="su_prev",
              errorbar="se",
              legend="full",
              ax=ax[0, 1])
ax[0, 0].set_title("blocked")
ax[0, 1].set_title("interleaved")
sns.move_legend(ax[0, 0], loc="upper left", bbox_to_anchor=(0.0, 1.0))
sns.move_legend(ax[0, 1], loc="upper left", bbox_to_anchor=(0.0, 1.0))
plt.savefig("../figures/summary_anova_emv.png")

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 4))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
sns.pointplot(data=dd[dd["condition"] == "blocked"],
              x="phase_2",
              y="delta_emv",
              hue="su_prev",
              errorbar="se",
              legend="full",
              ax=ax[0, 0])
sns.pointplot(data=dd[dd["condition"] == "interleaved"],
              x="phase_2",
              y="delta_emv",
              hue="su_prev",
              errorbar="se",
              legend="full",
              ax=ax[0, 1])
ax[0, 0].set_title("blocked")
ax[0, 1].set_title("interleaved")
sns.move_legend(ax[0, 0], loc="upper left", bbox_to_anchor=(0.0, 1.0))
sns.move_legend(ax[0, 1], loc="upper left", bbox_to_anchor=(0.0, 1.0))
plt.savefig("../figures/summary_anova_delta_emv.png")
