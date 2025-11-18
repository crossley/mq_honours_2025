from imports import *
from util_func import *

if __name__ == "__main__":

    dp = load_data()

    # subs_exc = []
    # dp = dp[~np.isin(dp["subject"], subs_exc)]

    fit_ss_model(dp)

    froot = "../fits/"

    fits_blocked_high_low = []
    for f in os.listdir(froot):
        if f.startswith("fit_results__cndBlocked - High Low_sub_"):
            d = pd.read_csv(os.path.join(froot, f), header=None)
            d.columns = ["alpha_low", "alpha_high", "beta_low", "beta_high", "sse"]
            d["condition"] = "Blocked - High Low"
            d["subject"] = int(f.split("_")[-1].split(".")[0])
            fits_blocked_high_low.append(d)

    fits_blocked_low_high = []
    for f in os.listdir(froot):
        if f.startswith("fit_results__cndBlocked - Low High_sub_"):
            d = pd.read_csv(os.path.join(froot, f), header=None)
            d.columns = ["alpha_low", "alpha_high", "beta_low", "beta_high", "sse"]
            d["condition"] = "Blocked - Low High"
            d["subject"] = int(f.split("_")[-1].split(".")[0])
            fits_blocked_low_high.append(d)

    fits_interleaved = []
    for f in os.listdir(froot):
        if f.startswith("fit_results__cndinterleaved_sub_"):
            d = pd.read_csv(os.path.join(froot, f), header=None)
            d.columns = ["alpha_low", "alpha_high", "beta_low", "beta_high", "sse"]
            d["condition"] = "interleaved"
            d["subject"] = int(f.split("_")[-1].split(".")[0])
            fits_interleaved.append(d)

    dblh = pd.concat(fits_blocked_low_high)
    dbhl = pd.concat(fits_blocked_high_low)
    di = pd.concat(fits_interleaved)

    dfits = pd.concat([dblh, dbhl, di]).reset_index(drop=True)

    # plot boxplot of alpha_low for the Blocked - Low High vs alpha_high for Blocked - High Low
    dfits["alpha"] = 0.0
    dfits.loc[dfits["condition"] == "Blocked - Low High",
              "alpha"] = dfits.loc[dfits["condition"] == "Blocked - Low High",
                                   "alpha_low"]
    dfits.loc[dfits["condition"] == "Blocked - High Low",
              "alpha"] = dfits.loc[dfits["condition"] == "Blocked - High Low",
                                   "alpha_high"]

#    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(6, 6))
#
#    alpha_blocked_low = dfits[dfits["condition"] == "Blocked - Low High"]["alpha_low"]
#    alpha_blocked_high = dfits[dfits["condition"] == "Blocked - High Low"]["alpha_high"]
#    beta_blocked_low = dfits[dfits["condition"] == "Blocked - Low High"]["beta_low"]
#    beta_blocked_high = dfits[dfits["condition"] == "Blocked - High Low"]["beta_high"]
#
#    alpha_interleaved_low = dfits[dfits["condition"] == "interleaved"]["alpha_low"]
#    alpha_interleaved_high = dfits[dfits["condition"] == "interleaved"]["alpha_high"]
#    beta_interleaved_low = dfits[dfits["condition"] == "interleaved"]["beta_low"]
#    beta_interleaved_high = dfits[dfits["condition"] == "interleaved"]["beta_high"]
#
#    ax[0, 0].boxplot([alpha_blocked_low, alpha_blocked_high], positions=[0, 1])
#    ax[0, 1].boxplot([beta_blocked_low, beta_blocked_high], positions=[0, 1])
#    ax[1, 0].boxplot([alpha_interleaved_low, alpha_interleaved_high], positions=[0, 1])
#    ax[1, 1].boxplot([beta_interleaved_low, beta_interleaved_high], positions=[0, 1])
#
#    # plot raw data points
#    ax[0, 0].scatter(np.zeros_like(alpha_blocked_low), alpha_blocked_low, color="k", alpha=0.5)
#    ax[0, 0].scatter(np.ones_like(alpha_blocked_high), alpha_blocked_high, color="k", alpha=0.5)
#    ax[0, 1].scatter(np.zeros_like(beta_blocked_low), beta_blocked_low, color="k", alpha=0.5)
#    ax[0, 1].scatter(np.ones_like(beta_blocked_high), beta_blocked_high, color="k", alpha=0.5)
#    ax[1, 0].scatter(np.zeros_like(alpha_interleaved_low), alpha_interleaved_low, color="k", alpha=0.5)
#    ax[1, 0].scatter(np.ones_like(alpha_interleaved_high), alpha_interleaved_high, color="k", alpha=0.5)
#    ax[1, 1].scatter(np.zeros_like(beta_interleaved_low), beta_interleaved_low, color="k", alpha=0.5)
#    ax[1, 1].scatter(np.ones_like(beta_interleaved_high), beta_interleaved_high, color="k", alpha=0.5)
#
#    ax[0, 0].set_xticklabels(["Low", "High"])
#    ax[0, 1].set_xticklabels(["Low", "High"])
#    ax[1, 0].set_xticklabels(["Low", "High"])
#    ax[1, 1].set_xticklabels(["Low", "High"])
#
#    ax[0, 0].set_ylabel("Alpha")
#    ax[0, 1].set_ylabel("Beta")
#    ax[1, 0].set_ylabel("Alpha")
#    ax[1, 1].set_ylabel("Beta")
#
#    ax[0, 0].set_title("Blocked Condition")
#    ax[0, 1].set_title("Blocked Condition")
#    ax[1, 0].set_title("Interleaved Condition")
#    ax[1, 1].set_title("Interleaved Condition")
#
#    plt.show()
#
#    print(pg.ttest(alpha_blocked_low, alpha_blocked_high, paired=False))
#    print(pg.ttest(beta_blocked_low, beta_blocked_high, paired=False))
#    print(pg.ttest(alpha_interleaved_low, alpha_interleaved_high, paired=True))
#    print(pg.ttest(beta_interleaved_low, beta_interleaved_high, paired=True))
#
#    # print non-parametric test results
#    print(pg.mwu(alpha_blocked_low, alpha_blocked_high))
#    print(pg.mwu(beta_blocked_low, beta_blocked_high))
#    print(pg.wilcoxon(alpha_interleaved_low, alpha_interleaved_high))
#    print(pg.wilcoxon(beta_interleaved_low, beta_interleaved_high))
#
#    cnds = dfits["condition"].unique()
#    d_pred_list = []
#    for j, c in enumerate(cnds):
#        dc = dp[dp["condition"] == c]
#        for s in dc["subject"].unique():
#            ds = dc[dc["subject"] == s]
#            ds = ds[ds["phase"].isin([2, 3])]
#            ds = ds.iloc[:225, :].reset_index(drop=True)
#            fs = dfits[(dfits["subject"] == s) & (dfits["condition"] == c)]
#            alpha_low = fs["alpha_low"].values[0]
#            alpha_high = fs["alpha_high"].values[0]
#            beta_low = fs["beta_low"].values[0]
#            beta_high = fs["beta_high"].values[0]
#
#            su = ds["su"].values
#            phase = ds["phase"].values
#            r = ds["rotation"].values
#            x_obs = ds["emv"].values
#
#            params = (alpha_low, alpha_high, beta_low, beta_high)
#            args = (r, x_obs, su, phase)
#
#            d_pred = sim_func(params, args)[1]
#            d_pred = pd.DataFrame(d_pred, columns=["emv"])
#            d_pred["trial"] = np.arange(d_pred.shape[0])
#            d_pred["subject"] = s
#            d_pred["condition"] = c
#            d_pred_list.append(d_pred)
#
#    d_pred_all = pd.concat(d_pred_list).reset_index(drop=True)
#    fig, ax = plt.subplots(1, len(cnds), squeeze=False, figsize=(8 * len(cnds), 6))
#    for j, c in enumerate(cnds):
#        dc = d_pred_all[d_pred_all["condition"] == c]
#        for s in dc["subject"].unique():
#            ds = dc[dc["subject"] == s]
#            ax[0, j].plot(ds["trial"], ds["emv"], alpha=0.5)
#        ax[0, j].set_title(c)
#        ax[0, j].set_xlabel("Trial")
#        ax[0, j].set_ylabel("Predicted EMV")
#
#    plt.show()

