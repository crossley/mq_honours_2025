from imports import *
from util_func import *

d, dd, ddd = load_data()

d["emv_rel"] = d["emv"] - d["target_angle"]
dd["emv_rel"] = d["emv"] - dd["target_angle"]
ddd["emv_rel"] = -d["emv"] - ddd["target_angle"]

d["target_angle"] = d["target_angle"].astype("category")
dd["target_angle"] = dd["target_angle"].astype("category")
ddd["target_angle"] = ddd["target_angle"].astype("category")

d.groupby(["condition"])["subject"].unique()
d.groupby(["condition"])["subject"].nunique()

for sub in d.subject.unique():

    ds = d[(d["subject"] == sub)][[
        "session", "phase", "trial", "emv_rel", "target_angle", "rotation"
    ]].drop_duplicates().sort_values("target_angle").reset_index(drop=True)

    dds = dd[(dd["subject"] == sub)][[
        "session", "phase", "trial", "emv_rel", "target_angle"
    ]].drop_duplicates()

    ddds = ddd[(ddd["subject"] == sub)][[
        "session", "phase", "relsamp", "t", "x", "y", "target_angle"
    ]].drop_duplicates()

    # TODO: use this plot to reject outlier trials
    # fig, ax = plt.subplots(3, 4, squeeze=False, figsize=(12, 8))
    # ax = ax.flatten()

    # for i, ta in enumerate(ds.target_angle.unique()):
    #     dta = ds[ds["target_angle"] == ta]
    #     dta = dta[dta["phase"] == "baseline"]
    #     sns.histplot(data=dta, x="emv_rel", bins=50, legend=False, ax=ax[i])
    #     ax[i].set_title("Target Angle: " + str(ta))
    # fig.suptitle("Subject " + str(sub))
    # plt.tight_layout()
    # plt.show()

    dp = ds.groupby(["session", "phase", "trial", "target_angle", "rotation"],
                    observed=True)["emv_rel"].mean().reset_index()

    dp.sort_values(["session", "trial"], inplace=True)

    dpp = ddds[ddds["phase"] == "generalization"].groupby(
        ["session", "phase", "target_angle", "relsamp"],
        observed=True)[["x", "y"]].mean().reset_index()
    dpp["training_target"] = False
    dpp.loc[dpp["target_angle"] == 0, "training_target"] = True
    dpp["training_target"] = dpp["training_target"].astype("category")

    dpp_base = ddds[ddds["phase"] == "baseline"].groupby(
        ["session", "phase", "target_angle",
         "relsamp"])[["x", "y"]].mean().reset_index()
    dpp_base["training_target"] = False
    dpp_base.loc[dpp_base["target_angle"] == 0, "training_target"] = True
    dpp_base["training_target"] = dpp_base["training_target"].astype(
        "category")

    dpg = ds[ds["phase"] == "generalization"].groupby(
        ["session", "target_angle"],
        observed=True)["emv_rel"].mean().reset_index()

    # NOTE: main results figure
    fig, ax = plt.subplots(4, 4, squeeze=False)
    fig.set_size_inches(16, 11)
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    fig.subplots_adjust(left=0.1, right=0.85)
    fig.subplots_adjust(top=0.9, bottom=0.1)

    for i, s in enumerate(dp.session.unique()):

        # initial movement vector across trials
        sns.scatterplot(
            data=dp[dp["session"] == s],
            x="trial",
            y="emv_rel",
            hue="target_angle",
            markers=True,
            legend=False,
            ax=ax[i, 0],
        )

        ax[i, 0].plot(dp[dp["session"] == s].trial,
                      dp[dp["session"] == s].rotation * 2,
                      color="black",
                      linestyle="-")

        ax[i, 0].set_ylim(-100, 100)
        ax[i, 0].set_title("Session " + str(s))
        ax[i, 0].set_ylabel("Endpoint Movement Vector")

        # generalisation function
        sns.barplot(data=dpg[(dpg["session"] == s)],
                    x="target_angle",
                    y="emv_rel",
                    errorbar="se",
                    hue="target_angle",
                    legend=False,
                    ax=ax[i, 1])
        ax[i, 1].set_xlabel("Target Angle")
        ax[i, 1].set_ylabel("Endpoint Movement Vector")
        for label in ax[i, 1].get_xticklabels():
            label.set_rotation(45)

        # bseline trajectories
        sns.scatterplot(
            data=dpp_base[dpp_base["session"] == s],
            x="x",
            y="y",
            hue="target_angle",
            style="training_target",
            markers=True,
            ax=ax[i, 2],
        )
        for ta in dpp.target_angle.unique():
            ta = ta + 90
            ax[i, 2].plot([0, 200 * np.cos(ta * np.pi / 180)],
                          [0, 200 * np.sin(ta * np.pi / 180)],
                          color="black",
                          linestyle="--")

        # generalisation trajectories
        sns.scatterplot(
            data=dpp[dpp["session"] == s],
            x="x",
            y="y",
            hue="target_angle",
            style="training_target",
            markers=True,
            ax=ax[i, 3],
        )
        for ta in dpp.target_angle.unique():
            ta = ta + 90
            ax[i, 3].plot([0, 200 * np.cos(ta * np.pi / 180)],
                          [0, 200 * np.sin(ta * np.pi / 180)],
                          color="black",
                          linestyle="--")

    [ax_.legend().remove() for ax_ in ax[:, 2]]
    [ax_.legend().remove() for ax_ in ax[:, 3]]
    ax[0, 2].legend(bbox_to_anchor=(2.5, 1), loc=2, borderaxespad=0.)
    fig.suptitle("Subject " + str(sub))
    plt.savefig("../figures/subject_" + str(sub) + ".png")
    plt.close()

# NOTE: visual inspection of the above figures reveals
# different qualitative patterns of generalisation across
# different subjects. Our interest is in the generalisation
# patters most plausibly related to implicit adaptation.
subs_gen_imp = [1, 4, 5, 6, 7, 8]

d = d[np.isin(d["subject"], subs_gen_imp)]
dd = dd[np.isin(dd["subject"], subs_gen_imp)]
ddd = ddd[np.isin(ddd["subject"], subs_gen_imp)]

d.groupby(["condition"])["subject"].unique()
d.groupby(["condition"])["subject"].nunique()

# NOTE: We also restict our attention to the targets in the
# neighborhood of the training target
d = d[np.isin(d["target_angle"], [-30, 0, 30])]
dd = dd[np.isin(dd["target_angle"], [-30, 0, 30])]
ddd = ddd[np.isin(ddd["target_angle"], [-30, 0, 30])]

d["target_angle"] = d["target_angle"].cat.remove_unused_categories()
dd["target_angle"] = dd["target_angle"].cat.remove_unused_categories()
ddd["target_angle"] = ddd["target_angle"].cat.remove_unused_categories()

from scipy.optimize import curve_fit


def gauss(x, a, b, c):
    return a * np.exp(-(x - 0)**2 / (2 * b**2)) + c


def fit_gaussian(x):

    ta = x["target_angle"].to_numpy()
    emv_rel = x["emv_rel"].to_numpy()

    popt, _ = curve_fit(gauss,
                        ta,
                        emv_rel,
                        p0=[1, 10, 1],
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

    x["popt_a"] = popt[0]
    x["popt_b"] = popt[1]
    x["popt_c"] = popt[2]

    return x


dp = d.groupby(["subject", "session", "phase", "trial", "target_angle"],
               observed=True)["emv_rel"].mean().reset_index()
dp.to_csv("../data_summary/data_by_trial.csv", index=False)

dpg = d[d["phase"] == "generalization"].groupby(
    ["subject", "session", "trial", "target_angle"],
    observed=True)["emv_rel"].mean().reset_index()

dpg["subject"] = dpg["subject"].astype("category")
d_fit = dpg.groupby(["subject", "session"]).apply(fit_gaussian)
d_fit = d_fit[["subject", "session", "popt_a", "popt_b",
               "popt_c"]].drop_duplicates()
d_fit.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
sns.barplot(data=d_fit, x="session", y="popt_a", errorbar="se", ax=ax[0, 0])
sns.barplot(data=d_fit, x="session", y="popt_b", errorbar="se", ax=ax[0, 1])
sns.barplot(data=d_fit, x="session", y="popt_c", errorbar="se", ax=ax[0, 2])
plt.savefig("../figures/gauss_fits.png")
plt.close()

d_fit.to_csv("../data_summary/gauss_fits.csv", index=False)

fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12, 8))
ax = ax.flatten()
for i, s in enumerate(dpg.session.unique()):
    sns.barplot(data=dpg[dpg["session"] == s],
                x="target_angle",
                y="emv_rel",
                errorbar="se",
                hue="subject",
                ax=ax[i])
plt.savefig("../figures/generalization.png")
plt.close()
