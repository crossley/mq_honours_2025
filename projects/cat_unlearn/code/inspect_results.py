import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower, TTestPower

# NOTE: power analysis for sample size
analysis = TTestPower()

effect_size = 1.0
alpha = 0.05
power = 0.8
sample_size_per_group = analysis.solve_power(effect_size=effect_size,
                                             power=power,
                                             alpha=alpha,
                                             alternative='two-sided')

print(f"Required sample size per group: {sample_size_per_group:.2f}")

# NOTE: inspect results
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

        # fig, ax = plt.subplots(1, 4, squeeze=False, figsize=(12, 6))
        # sns.scatterplot(data=d[d["phase"] == "Learn"],
        #                 x="x",
        #                 y="y",
        #                 hue="cat",
        #                 ax=ax[0, 0])
        # sns.scatterplot(data=d[d["phase"] == "Intervention"],
        #                 x="x",
        #                 y="y",
        #                 hue="cat",
        #                 ax=ax[0, 1])
        # sns.scatterplot(data=d[d["phase"] == "Test"],
        #                 x="x",
        #                 y="y",
        #                 hue="cat",
        #                 ax=ax[0, 2])
        # sns.lineplot(data=d.groupby(["phase", "block"])[["acc"]].mean(),
        #              x="block",
        #              y="acc",
        #              hue="phase",
        #              ax=ax[0, 3])
        # ax[0, 0].set_title(d.subject[0])
        # ax[0, 1].set_title(d.experiment[0])
        # ax[0, 2].set_title(d.condition[0])
        # plt.tight_layout()
        # plt.show()

d = pd.concat(d_rec, ignore_index=True)

print(d.groupby(["experiment", "condition"])["subject"].unique())
print(d.groupby(["experiment", "condition"])["subject"].nunique())

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
# dd[(dd["experiment"] == 1)].to_csv("../data_summary/summary.csv", index=False)

import statsmodels.formula.api as smf
import statsmodels.api as sm

d['trial'] = d['trial'] + 1
d['acc'] = d['acc'].astype(int)
d['condition'] = d['condition'].astype('category')
d['phase'] = d['phase'].astype('category')

d['log_trial'] = np.log(d['trial'])

d['condition'] = d['condition'].cat.reorder_categories(
    ['relearn', 'new_learn'])
d['phase'] = d['phase'].cat.reorder_categories(
    ['Learning', 'Intervention', 'Test'])

model = smf.glm(
    formula='acc ~ log_trial + condition + phase + condition:phase',
    data=d,
    family=sm.families.Binomial()).fit()

print(model.summary())
