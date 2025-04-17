import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

dir_data = "../data"

d_rec = []

for f in os.listdir(dir_data):
    if f.endswith(".csv"):
        d = pd.read_csv(os.path.join(dir_data, f))
        d_rec.append(d)

d = pd.concat(d_rec)

# pick out subset of subjects: 999, 1000, 1001, 1002
d = d[d["subject"].isin([999, 1000, 1001, 1002])]

print(d.groupby(["condition"])["subject"].unique())
print(d.groupby(["condition"])["subject"].nunique())

d["acc"] = d["cat"] == d["resp"]

d["cat"] = d["cat"].astype("category")
d["sub_task"] = d["sub_task"].astype("category")

# recode cat level names
d["cat"] = d["cat"].cat.rename_categories({
    102: "L1",  # 'f'
    106: "R1",  # 'j'
    100: "L2",  # 'd'
    107: "R2",  # 'k'
    99:  "L3",  # 'c'
    109: "R3",  # 'm'
    113: "L4",  # 'q'
    112: "R4",  # 'p'
})

# reorder the levels of the cat variable L1, R1, L2, R2, L3, R3, L4, R4
d["cat"] = d["cat"].cat.reorder_categories([
    "L1", "R1", "L2", "R2", "L3", "R3", "L4", "R4"
])

d["condition"] = d["condition"].astype("category")

fig, ax = plt.subplots(d["sub_task"].nunique(),
                       d["condition"].nunique(),
                       squeeze=False,
                       figsize=(16, 8))
for cnd in d["condition"].cat.categories:
    dc = d[d["condition"] == cnd].copy()
    for st in dc["sub_task"].unique():
        dcst = dc[dc["sub_task"] == st].copy()
        dcst["cat"] = dcst["cat"].cat.remove_unused_categories()
        row_ind = dc["sub_task"].cat.categories.get_loc(st)
        col_ind = dc["condition"].cat.categories.get_loc(cnd)
        axx = ax[row_ind, col_ind]
        sns.scatterplot(data=dcst,
                        x="x",
                        y="y",
                        hue="cat",
                        style="cat",
                        ax=axx)
        sns.move_legend(axx, "upper left")
        axx.set_title("%s, sub_task: %s" % (cnd, st))
plt.tight_layout()
plt.savefig("../figures/fig_categories_stim_space.png")
plt.close()

# add a block column that split trials up into blocks of 25
d["block"] = np.floor(d["trial"] / 25).astype(int)

# calculate accuracy for each block
dd = d.groupby(["condition", "subject", "sub_task", "block"],
               observed=True)["acc"].mean().reset_index()

fig, ax = plt.subplots(1,
                       d["condition"].nunique(),
                       squeeze=False,
                       figsize=(16, 8))
for i, cnd in enumerate(d["condition"].cat.categories):
    dc = d[d["condition"] == cnd].copy()
    sns.lineplot(data=dc,
                 x="block",
                 y="acc",
                 style="sub_task",
                 ax=ax[0, i])
    ax[0, i].set_ylim(0, 1)
    ax[0, i].set_title(cnd)
    sns.move_legend(ax[0, i], "upper left")
plt.tight_layout()
plt.savefig("../figures/fig_accuracy_per_block.png")
plt.close()


## print the unique subjects in each condition
print(d.groupby(["condition"])["subject"].unique())

# dd = dd.sort_values(
#     by=["condition", "subject", "sub_task", "block"]).reset_index(drop=True)
# 
# dd.to_csv("../data_summary/summary.csv", index=False)
# 
# # NOTE: stats
# dd["block"] = dd["block"].astype("category")
# dd["condition"] = dd["condition"].astype("category")
# dd["sub_task"] = dd["sub_task"].astype("category")
# 
# dd.groupby(["condition"])["subject"].nunique()
# 
# pg.mixed_anova(data=dd,
#                dv="acc",
#                subject="subject",
#                within="block",
#                between="condition")
# 
# # NOTE: use stats models to perform a logistc regression
# # using `d` as the data frame, `acc` as the observed
# # variable, `trial` as discrete predictor, `condition` as a
# # categorical predictor, and `sub_task` as a categorical
# # predictor. The model should be fit to the data using
# # a binomial distribution.
# import statsmodels.api as sm
# import patsy
# 
# dd = d[["trial", "condition", "sub_task", "acc"]].copy()
# dd["intercept"] = 1
# dd = pd.get_dummies(dd,
#                     columns=["condition", "sub_task"],
#                     drop_first=True,
#                     dtype=int)
# 
# dd = dd.rename(columns={"condition_4F4K_incongruent": "condition"})
# dd = dd.rename(columns={"sub_task_2": "sub_task"})
# 
# endog = dd["acc"]
# exog = patsy.dmatrix("np.log(trial) * condition * sub_task",
#                      data=dd,
#                      return_type="dataframe")
# 
# model = sm.GLM(endog, exog, family=sm.families.Binomial())
# fm = model.fit()
# print(fm.summary())
# 
# # NOTE: plot the predicted probabilities
# dd["pred"] = fm.predict(exog)
# 
# fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
# sns.lineplot(data=dd,
#              x="trial",
#              y="acc",
#              hue="condition",
#              alpha=0.5,
#              legend=False,
#              ax=ax[0, 0])
# sns.lineplot(data=dd, x="trial", y="pred", hue="condition", ax=ax[0, 0])
# ax[0, 0].set_ylim(0, 1)
# plt.savefig("../figures/fig_logistic_regression.png")
# plt.close()
