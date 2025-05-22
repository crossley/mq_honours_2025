import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
import patsy

dir_data = "../data"

d_rec = []

for f in os.listdir(dir_data):
    if f.endswith("data.csv"):  # NOTE: exludes "move" files
        d = pd.read_csv(os.path.join(dir_data, f))
        d.sort_values(by=["condition", "subject", "trial"], inplace=True)
        d["acc"] = d["cat"] == d["resp"]
        d["cat"] = d["cat"].astype("category")
        d["trial"] += 1
        dd = d[["trial", "condition", "acc"]].copy()
        dd["intercept"] = 1
        endog = dd["acc"]
        exog = patsy.dmatrix("np.log(trial)",
                             data=dd,
                             return_type="dataframe")
        model = sm.GLM(endog, exog, family=sm.families.Binomial())
        fm = model.fit()
        d["pred"] = fm.predict(exog)
        d_rec.append(d)

d = pd.concat(d_rec)

print(d.groupby(["condition"])["subject"].unique())
print(d.groupby(["condition"])["subject"].nunique())

# for s in d["subject"].unique():
#     ds = d[d["subject"] == s]
#
#     # plot x and y coordinates
#     fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
#     sns.scatterplot(data=ds,
#                     x="x",
#                     y="y",
#                     hue="cat",
#                     style="cat",
#                     ax=ax[0, 0])
#     sns.scatterplot(data=ds,
#                     x="xt",
#                     y="yt",
#                     hue="cat",
#                     style="cat",
#                     ax=ax[0, 1])
#     plt.show()
#
#     # print(s, ds.target_angle_left.unique(), ds.shape, ds.condition.unique())
#     # print(s, ds.target_angle_right.unique(), ds.shape, ds.condition.unique())
#
#     # ds = d[d["subject"] == s]
#     # ds.plot(subplots=True)
#     # plt.show()

fig, ax = plt.subplots(d["subject"].unique().shape[0], 1, squeeze=False, figsize=(12, 8))
ax = ax.flatten()
for i, s in enumerate(d["subject"].unique()):
    axx = ax[i]
    ds = d[d["subject"] == s]
    sns.lineplot(data=ds, x="trial", y="acc", ax=axx)
    sns.lineplot(data=ds, x="trial", y="pred", ax=axx)
    axx.set_title(f"Subject {s} - {ds.condition.unique()[0]}")
plt.tight_layout()
plt.show()
