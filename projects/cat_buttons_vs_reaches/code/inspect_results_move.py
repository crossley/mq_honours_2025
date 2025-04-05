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
    x = d["xx"]
    y = d["yy"]
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
    x = d["xx"].to_numpy()
    y = d["yy"].to_numpy()

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

for s in [2]:

    f_trl = "sub_{}_data.csv".format(s)
    f_mv = "sub_{}_data_move.csv".format(s)

    d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
    d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

    d_trl = d_trl.sort_values(["condition", "subject", "trial"])
    d_mv = d_mv.sort_values(["condition", "subject", "t", "trial"])

    d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
    x_start = d_hold.xx.mean()
    y_start = d_hold.yy.mean()

    d_mv = d_mv[d_mv["state"].isin(["state_moving"])]

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
d["yy"] = -d["yy"]

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 6))
sns.scatterplot(data=d,
                x="xx",
                y="yy",
                hue="trial",
                ax=ax[0, 0],
                )
plt.show()
