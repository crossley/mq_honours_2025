from imports import *
from util_func import *

if __name__ == "__main__":

    dp = load_data()

    print(dp)

    # fit_ss_model(dp)

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

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(6, 6))

    alpha_blocked_low = dfits[dfits["condition"] == "Blocked - Low High"]["alpha_low"]
    alpha_blocked_high = dfits[dfits["condition"] == "Blocked - High Low"]["alpha_high"]
    beta_blocked_low = dfits[dfits["condition"] == "Blocked - Low High"]["beta_low"]
    beta_blocked_high = dfits[dfits["condition"] == "Blocked - High Low"]["beta_high"]

    alpha_interleaved_low = dfits[dfits["condition"] == "interleaved"]["alpha_low"]
    alpha_interleaved_high = dfits[dfits["condition"] == "interleaved"]["alpha_high"]
    beta_interleaved_low = dfits[dfits["condition"] == "interleaved"]["beta_low"]
    beta_interleaved_high = dfits[dfits["condition"] == "interleaved"]["beta_high"]

    print(pg.ttest(alpha_blocked_low, alpha_blocked_high, paired=False))
    print(pg.ttest(beta_blocked_low, beta_blocked_high, paired=False))
    print(pg.ttest(alpha_interleaved_low, alpha_interleaved_high, paired=True))
    print(pg.ttest(beta_interleaved_low, beta_interleaved_high, paired=True))

    ax[0, 0].boxplot([alpha_blocked_low, alpha_blocked_high])
    ax[0, 1].boxplot([beta_blocked_low, beta_blocked_high])
    ax[1, 0].boxplot([alpha_interleaved_low, alpha_interleaved_high])
    ax[1, 1].boxplot([beta_interleaved_low, beta_interleaved_high])

    ax[0, 0].set_xticklabels(["Low", "High"])
    ax[0, 1].set_xticklabels(["Low", "High"])
    ax[1, 0].set_xticklabels(["Low", "High"])
    ax[1, 1].set_xticklabels(["Low", "High"])

    ax[0, 0].set_ylabel("Alpha")
    ax[0, 1].set_ylabel("Beta")
    ax[1, 0].set_ylabel("Alpha")
    ax[1, 1].set_ylabel("Beta")

    ax[0, 0].set_title("Blocked Condition")
    ax[0, 1].set_title("Blocked Condition")
    ax[1, 0].set_title("Interleaved Condition")
    ax[1, 1].set_title("Interleaved Condition")

    plt.show()

    df_long = dfits.melt(id_vars="condition",
                         value_vars=["alpha_low", "alpha_high", "beta"],
                         var_name="parameter",
                         value_name="value")

    conditions = df_long["condition"].unique()
    fig, axes = plt.subplots(1,
                             len(conditions),
                             figsize=(5 * len(conditions), 6),
                             sharey=True)

    for ax, cond in zip(axes, conditions):
        sns.boxplot(data=df_long[df_long["condition"] == cond],
                    x="parameter",
                    y="value",
                    ax=ax)
        ax.set_title(f"Condition: {cond}")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.show()

    # generate predicted data for each subject and condition
    for c in dfits["condition"].unique():
        d_pred_list = []
        dc = dp[dp["condition"] == c]
        for s in dc["subject"].unique():
            ds = dc[dc["subject"] == s]
            ds = ds.iloc[:424, :]
            fs = dfits[(dfits["subject"] == s) & (dfits["condition"] == c)]
            alpha_low = fs["alpha_low"].values[0]
            alpha_high = fs["alpha_high"].values[0]
            beta = fs["beta"].values[0]

            su = ds["su"].values
            r = ds["rotation"].values
            x_obs = ds["emv"].values

            params = (alpha_low, alpha_high, beta)
            args = (r, x_obs, su)

            d_pred = sim_func(params, args)[1]

            # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 6))
            # ax[0, 0].plot(x_obs, color="k", alpha=0.5, label="observed")
            # ax[0, 0].plot(d_pred, color="r", alpha=0.5, label="predicted")
            # ax[0, 0].set_title(f"{c} - sub {s}")
            # ax[0, 0].legend()
            # plt.show()

            d_pred_list.append(d_pred)

        d_pred = np.vstack(d_pred_list)

        # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 6))
        # for i in range(d_pred.shape[0]):
        #     ax[0, 0].plot(d_pred[i, :], color="k", alpha=0.1)
        # ax[0, 0].set_title(c)
        # plt.show()
