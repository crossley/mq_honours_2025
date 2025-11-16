from imports import *

def fit_ss_model(d):

    froot = "../fits/"

    # alpha_low, alpha_high, beta
    bounds = (
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    )

    # constraints = LinearConstraint(
    #     A=[
    #         [ 0, 0, 0, ],
    #         [ 0, 0, 0, ],
    #         [ 0, 0, 0, ],
    #     ],
    #     lb=[ 0, 0, 0, ],
    #     ub=[ 0, 0, 0, ],
    # )

    # to improve your chances of finding a global minimum use higher
    # popsize (default 15), with higher mutation (default 0.5) and
    # (dithering), but lower recombination (default 0.7). this has the
    # effect of widening the search radius, but slowing convergence.
    fit_args = {
        "obj_func": obj_func,
        "sim_func": sim_func,
        "bounds": bounds,
        # "constraints": constraints,
        "disp": False,
        "maxiter": 3000,
        "popsize": 22,
        "mutation": 0.8,
        "recombination": 0.4,
        "tol": 1e-3,
        "polish": True,
        "updating": "deferred",
        "workers": -1,
    }

    for cnd in d["condition"].unique():

        dcnd = d[d["condition"] == cnd]

        for sub in dcnd["subject"].unique():

            dsub = dcnd[dcnd["subject"] == sub].copy()
            dsub = dsub[dsub["phase"].isin([1, 2, 3, 4])]
            dsub = dsub[["rotation", "emv", "trial", "su", "phase"]]

            rot = dsub["rotation"].to_numpy()
            x_obs = dsub["emv"].to_numpy()
            su = dsub["su"].to_numpy()
            phase = dsub["phase"].to_numpy()

            args = (rot, x_obs, su, phase)

            results = differential_evolution(
                func=fit_args["obj_func"],
                bounds=fit_args["bounds"],
                # constraints=fit_args["constraints"],
                args=args,
                disp=fit_args["disp"],
                maxiter=fit_args["maxiter"],
                popsize=fit_args["popsize"],
                mutation=fit_args["mutation"],
                recombination=fit_args["recombination"],
                tol=fit_args["tol"],
                polish=fit_args["polish"],
                updating=fit_args["updating"],
                workers=fit_args["workers"],
            )

            # pe = results["x"]
            # x_pred = sim_func(pe, args)[1]
            # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 8))
            # ax[0, 0].plot(x_obs, label='observed')
            # ax[0, 0].plot(x_pred, label='predicted')
            # ax[0, 0].legend()
            # plt.show()

            fout = os.path.join(
                froot,
                "fit_results_" + "_cnd" + str(cnd) + "_sub_" + str(sub) +
                ".txt",
            )
            with open(fout, "w") as f:
                tmp = np.concatenate((results["x"], [results["fun"]]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, "%0.4f", delimiter=",", newline="\n")


def sim_func(params, args):

    alpha_low = params[0]
    alpha_high = params[1]
    beta_low = params[2]
    beta_high= params[3]

    r = args[0]
    x_obs = args[1]
    su = args[2]
    phase = args[3]

    n_trials = r.shape[0]

    delta = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    yff = np.zeros(n_trials)

    for i in range(n_trials - 1):

        yff[i] = xff[i]

        if phase[i] != 3:
            delta[i] = yff[i] + r[i]
        else:
            delta[i] = 0.0

        if su[i] == "low":
            xff[i + 1] = beta_low * xff[i] + alpha_low * delta[i]
        else:
            xff[i + 1] = beta_high * xff[i] + alpha_high * delta[i]

    return (yff, xff)


def obj_func(params, *args):
    obs = args

    rot = obs[0]
    x_obs = obs[1]
    su = obs[2]
    phase = obs[3]

    args = (rot, x_obs, su, phase)

    x_pred = sim_func(params, args)[0]

    sse = np.sum((x_obs - x_pred)**2)

    return sse

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

def load_data():

    dir_data = "../data/"

    d_rec = []

    # iterate over files in the ../data directory
    for f in os.listdir(dir_data):

        if f.endswith(".csv"):

            # extract subject number
            s = int(f.split("_")[1])

            # try to load both trial and movement files
            try:
                f_trl = "sub_{}_data.csv".format(s)
                f_mv = "sub_{}_data_move.csv".format(s)

                d_trl = pd.read_csv(os.path.join(dir_data, f_trl))
                d_mv = pd.read_csv(os.path.join(dir_data, f_mv))

                if d_trl.shape[0] != 429:
                    print("Subject {} has anomolous trial data".format(s))

                else:
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

            # print warning if file load fails
            except Exception as e:
                print("Could not load data for subject {}: {}".format(s, e))

    d = pd.concat(d_rec)

    d["su"] = d["su"].cat.rename_categories({0.0: "low", 26.78: "high"})

    d.groupby(["condition"])["subject"].unique()
    d.groupby(["condition"])["subject"].nunique()

    d.sort_values(["condition", "subject", "trial", "t"], inplace=True)

    for s in d["subject"].unique():
        ds = d[d["subject"] == s]
        if ds["condition"].unique() == "blocked":
            if ds[ds["phase"] == 2]["su"].unique() == "low":
                d.loc[d["subject"] == s, "condition"] = "Blocked - Low High"
            else:
                d.loc[d["subject"] == s, "condition"] = "Blocked - High Low"

    d.groupby(["condition"])["subject"].unique()
    d.groupby(["condition"])["subject"].nunique()
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

    return dp
