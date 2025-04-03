import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import seaborn as sns

dir_data = "../data"

d_rec = []

for file in os.listdir(dir_data):

    if file.endswith(".csv"):
        d = pd.read_csv(os.path.join(dir_data, file))
        d["block"] = np.floor(d["trial"] / 25).astype(int)
        d["acc"] = d["cat"] == d["resp"]
        d["cat"] = d["cat"].astype("category")
        d["phase"] = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"] * 299
        d_rec.append(d)

        fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(12, 6))
        sns.scatterplot(data=d[d["phase"] == "Learn"],
                        x="x",
                        y="y",
                        hue="cat",
                        ax=ax[0, 0])
        sns.scatterplot(data=d[d["phase"] == "Intervention"],
                        x="x",
                        y="y",
                        hue="cat",
                        ax=ax[0, 1])
        sns.scatterplot(data=d[d["phase"] == "Test"],
                        x="x",
                        y="y",
                        hue="cat",
                        ax=ax[0, 2])
        sns.lineplot(data=d.groupby(["phase", "block"])[["acc"]].mean(),
                     x="block",
                     y="acc",
                     hue="phase",
                     ax=ax[0, 3])
        ax[0, 0].set_title(d.subject[0])
        ax[0, 1].set_title(d.experiment[0])
        ax[0, 2].set_title(d.condition[0])
        plt.tight_layout()
        plt.show()

d = pd.concat(d_rec, ignore_index=True)

# calculate accuracy for each block
dd = d.groupby(["experiment", "condition", "subject", "block",
                "phase"])["acc"].mean().reset_index()

# 

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.lineplot(data=dd[(dd["experiment"] == 1)],
             x="block",
             y="acc",
             hue="condition",
             style="phase",
             legend=True,
             ax=ax[0, 0])
ax[0, 0].set_title("Exp 1")
plt.tight_layout()
plt.show()

d.groupby(["experiment", "condition"])["subject"].unique()
d.groupby(["experiment", "condition"])["subject"].nunique()

dd = dd.sort_values(["experiment", "condition", "subject", "block", "phase"])
dd[(dd["experiment"] == 1)].to_csv("../data_summary/summary.csv", index=False)

# NOTE: begin backwards learning curve analysis
d = pd.concat(d_rec, ignore_index=True)
d = d[["experiment", "condition", "subject", "phase", "trial", "acc"]].copy()
d = d[d["experiment"] == 1]

thresh_blc =  0.85

def add_blc(x):

    b, a = butter(3, 0.01)
    x["acc_smooth"] = lfilter(b, a, x["acc"])
    idx = np.where(x["acc_smooth"] > thresh_blc * x["acc_smooth"].max())[0][0]
    x["blc_idx"] = idx
    x["trial_blc"] = x["trial"] - x["trial"].min() - idx
    return x

    # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
    # ds.acc_smooth.plot(ax=ax[0, 0])
    # ax[0, 0].axvline(x=idx[0], color="red", linestyle="--")
    # plt.ylim(0, 1)
    # plt.show()

d = d.groupby(["experiment", "condition", "subject", "phase"])[["trial", "acc"]].apply(add_blc)
d = d.reset_index()

fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(6, 6))
sns.lineplot(data=d[d["phase"]=="Learn"], x="trial_blc", y="acc_smooth", hue="condition", ax=ax[0, 0])
sns.lineplot(data=d[d["phase"]=="Test"], x="trial_blc", y="acc_smooth", hue="condition", ax=ax[0, 1])
ax[0, 0].set_ylim(0, 1)
ax[0, 1].set_ylim(0, 1)
ax[0, 0].set_title("Phase: Learn")
ax[0, 1].set_title("Phase: Test")
plt.show()
