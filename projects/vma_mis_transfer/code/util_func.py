from imports import *
from util_func import *


def compute_kinematics(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()

    x = x - x[0]
    y = y - y[0]
    y = -y

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    v = np.sqrt(vx**2 + vy**2)

    r = np.sqrt(x**2 + y**2)

    v_peak = v.max()
    # ts = t[v > (0.05 * v_peak)][0]
    ts = t[r > 0.1 * r.max()][0]

    # xx = x[(t >= ts) & (t <= ts + 0.1)].mean()
    # yy = y[(t >= ts) & (t <= ts + 0.1)].mean()

    xx = x[-1]
    yy = y[-1]

    theta = (np.arctan(yy / xx)) * 180 / np.pi

    if (xx > 0) & (yy > 0):
        theta = -theta + 90

    elif (xx > 0) & (yy < 0):
        theta = -(theta - 90)

    elif (xx < 0) & (yy > 0):
        theta = -theta - 90

    elif (xx < 0) & (yy < 0):
        theta = -theta - 90

    # TODO: currently ignoring imv
    # imv = theta[(t >= ts) & (t <= ts + 0.1)].mean()
    imv = theta
    emv = theta

    d["x"] = x
    d["y"] = y
    d["v"] = v
    d["imv"] = imv
    d["emv"] = emv

    return d


def interpolate_movements(d):
    t = d["t"].to_numpy()
    x = d["x"].to_numpy()
    y = d["y"].to_numpy()
    v = d["v"].to_numpy()

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
    dd["session"] = d["session"].unique()[0]
    dd["trial"] = d["trial"].unique()[0]
    dd["target_angle"] = d["target_angle"].unique()[0]
    dd["phase"] = d["phase"].unique()[0]
    dd["su"] = d["su"].unique()[0]
    dd["imv"] = d["imv"].unique()[0]
    dd["emv"] = d["emv"].unique()[0]

    return dd


def bootstrap_ci(x, n, alpha):
    x_boot = np.zeros(n)
    for i in range(n):
        x_boot[i] = np.random.choice(x, x.shape, replace=True).mean()
        ci = np.percentile(x_boot, [alpha / 2, 1.0 - alpha / 2])
    return (ci)


def bootstrap_t(x_obs, y_obs, x_samp_dist, y_samp_dist, n):
    d_obs = x_obs - y_obs

    d_boot = np.zeros(n)
    xs = np.random.choice(x_samp_dist, n, replace=True)
    ys = np.random.choice(y_samp_dist, n, replace=True)
    d_boot = xs - ys
    d_boot = d_boot - d_boot.mean()

    p_null = (1 + np.sum(np.abs(d_boot) > np.abs(d_obs))) / (n + 1)
    return (p_null)


def g_func_gauss(theta, theta_mu, sigma):
    if sigma != 0:
        G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
    else:
        G = np.zeros(11)
    return G


def g_func_flat(amp):
    G = amp * np.ones(11)
    return G


def simulate_state_space_with_g_func_2_state(p, rot):
    alpha_s = p[0]
    beta_s = p[1]
    g_sigma_s = p[2]
    alpha_f = p[3]
    beta_f = p[4]
    g_sigma_f = p[5]
    beta_s_2 = p[6]
    beta_f_2 = p[7]

    num_trials = rot.shape[0]

    delta = np.zeros(num_trials)

    theta_values = np.array(
        [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    theta_train_ind = np.where(theta_values == 0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    x = np.zeros((11, num_trials))
    xs = np.zeros((11, num_trials))
    xf = np.zeros((11, num_trials))
    for i in range(0, num_trials - 1):
        if np.isnan(rot[i]):
            delta[i] = 0.0
        else:
            delta[i] = x[theta_ind[i], i] - rot[i]

        Gs = g_func_gauss(theta_values, theta_values[theta_ind[i]], g_sigma_s)
        Gf = g_func_flat(g_sigma_f)

        if i < 341:
            xs[:, i + 1] = (1 - beta_s) * xs[:, i] - alpha_s * delta[i] * Gs
            xf[:, i + 1] = (1 - beta_f) * xf[:, i] - alpha_f * delta[i] * Gf

        elif i > 341:
            # xs[:, i + 1] = (1 - beta_s_2) * xs[:, i]
            # xf[:, i + 1] = (1 - beta_f_2) * xf[:, i]
            xs[:, i + 1] = xs[:, i]
            xf[:, i + 1] = xf[:, i]

        elif i == 341:
            # xs[:, i + 1] = (1 - beta_s_2) * xs[:, i]
            # xf[:, i + 1] = (1 - beta_f_2) * xf[:, i]
            # xs[:, i + 1] = xs[:, i]
            # xf[:, i + 1] = xf[:, i]
            xs[:, i + 1] = xs[:, i] * beta_s_2
            xf[:, i + 1] = xf[:, i] * beta_f_2

        x[:, i + 1] = xs[:, i + 1] + xf[:, i + 1]

    return (x.T, xs.T, xf.T)


def fit_obj_func_sse(params, *args):
    x_obs = args[0]
    rot = args[1]
    x_pred = simulate_state_space_with_g_func_2_state(params, rot)[0]

    sse_rec = np.zeros(11)
    for i in [3, 4, 5, 6, 7]:
        sse_rec[i] = np.nansum(
            (x_obs[:341, i] - x_pred[:341, i])**2) + 2 * np.nansum(
                (x_obs[341:, i] - x_pred[341:, i])**2)
        sse = np.nansum(sse_rec)
    return sse


def load_data():
    dir_data = "../data/"

    d_rec = []
    dd_rec = []
    ddd_rec = []

    sub_nums = [1, 3, 4, 5, 6, 7, 8, 9, 10]

    for s in sub_nums:

        f_trl_1 = "sub_{}_data.csv".format(s)
        f_mv_1 = "sub_{}_data_move.csv".format(s)

        f_trl_2 = "sub_{}{}_data.csv".format(s, s)
        f_mv_2 = "sub_{}{}_data_move.csv".format(s, s)

        f_trl_3 = "sub_{}{}{}_data.csv".format(s, s, s)
        f_mv_3 = "sub_{}{}{}_data_move.csv".format(s, s, s)

        f_trl_4 = "sub_{}{}{}{}_data.csv".format(s, s, s, s)
        f_mv_4 = "sub_{}{}{}{}_data_move.csv".format(s, s, s, s)

        d_trl_1 = pd.read_csv(os.path.join(dir_data, f_trl_1))
        d_mv_1 = pd.read_csv(os.path.join(dir_data, f_mv_1))

        d_trl_2 = pd.read_csv(os.path.join(dir_data, f_trl_2))
        d_mv_2 = pd.read_csv(os.path.join(dir_data, f_mv_2))

        d_trl_3 = pd.read_csv(os.path.join(dir_data, f_trl_3))
        d_mv_3 = pd.read_csv(os.path.join(dir_data, f_mv_3))

        d_trl_4 = pd.read_csv(os.path.join(dir_data, f_trl_4))
        d_mv_4 = pd.read_csv(os.path.join(dir_data, f_mv_4))

        d_trl_1["session"] = 1
        d_mv_1["session"] = 1

        d_trl_2["session"] = 2
        d_mv_2["session"] = 2

        d_trl_3["session"] = 3
        d_mv_3["session"] = 3

        d_trl_4["session"] = 4
        d_mv_4["session"] = 4

        d_trl_2["subject"] = d_trl_1["subject"].unique()[0]
        d_mv_2["subject"] = d_mv_1["subject"].unique()[0]

        d_trl_3["subject"] = d_trl_1["subject"].unique()[0]
        d_mv_3["subject"] = d_mv_1["subject"].unique()[0]

        d_trl_4["subject"] = d_trl_1["subject"].unique()[0]
        d_mv_4["subject"] = d_mv_1["subject"].unique()[0]

        d_trl = pd.concat([d_trl_1, d_trl_2, d_trl_3, d_trl_4])
        d_mv = pd.concat([d_mv_1, d_mv_2, d_mv_3, d_mv_4])

        d_trl = d_trl.sort_values(["condition", "subject", "session", "trial"])
        d_mv = d_mv.sort_values(
            ["condition", "subject", "session", "trial", "t"])
        d_hold = d_mv[d_mv["state"].isin(["state_holding"])]
        x_start = d_hold.x.mean()
        y_start = d_hold.y.mean()

        d_mv = d_mv[d_mv["state"].isin(["state_moving"])]

        phase = np.empty(d_trl.shape[0] // 4, dtype="object")

        # The experiment began with a familiarization phase of 33
        # reach trials (3 trials per target in pseudorandom order)
        # with continuous veridical visual feedback provided
        # throughout the reach.
        phase[:33] = "familiarisation"

        # The baseline phase consisted of 198 reach trials across
        # all 11 target directions (18 trials per target). On each
        # trial, the location of the target was randomized across
        # participants. For 2/3 of the reaches (132 trials),
        # continuous veridical cursor feedback was provided
        # throughout the trial. For the remaining 1/3 (66 trials),
        # visual feedback was completely withheld (i.e., no feedback
        # was given during the reach and no feedback was given at
        # the end of the reach about reach accuracy).
        phase[33:231] = "baseline"

        # The adaptation phase consisted of 110 reaches toward a
        # single target positioned at 0° in the frontal plane
        # (straight ahead; see Fig. 1b). During this phase, endpoint
        # feedback was rotated about the starting position by 30°
        # (CW or CCW; counterbalanced between participants).
        phase[231:341] = "adaptation"

        # The generalization phase consisted of 66 reaches to 1 of
        # 11 target directions (10 untrained directions) presented
        # in pseudorandom order without visual feedback.
        phase[341:] = "generalization"

        phase = np.concatenate([phase, phase, phase, phase])

        d_trl["phase"] = phase

        d_trl["su"] = d_trl["su"].astype("category")

        d_trl["ep"] = (d_trl["ep"] * 180 / np.pi) + 90
        d_trl["rotation"] = d_trl["rotation"] * 180 / np.pi

        d = pd.merge(d_mv,
                     d_trl,
                     how="outer",
                     on=["condition", "subject", "session", "trial"])

        d = d.groupby(["condition", "subject", "session", "trial"],
                      group_keys=False).apply(compute_kinematics)

        dd = d.groupby([
            "condition", "subject", "session", "phase", "target_angle", "trial"
        ],
                       group_keys=False).apply(interpolate_movements)

        ddd = (dd.groupby([
            "condition", "subject", "session", "phase", "target_angle",
            "relsamp"
        ])[["t", "x", "y", "v"]].mean().reset_index())

        d_rec.append(d)
        dd_rec.append(dd)
        ddd_rec.append(ddd)

    d = pd.concat(d_rec)
    dd = pd.concat(dd_rec)
    ddd = pd.concat(ddd_rec)

    d = d.reset_index(drop=True)
    dd = dd.reset_index(drop=True)
    ddd = ddd.reset_index(drop=True)

    return d, dd, ddd


def prep_for_fits():
    d, _, _ = load_data()

    d = d[[
        'session', 'subject', 'phase', 'trial', 'rotation', 'target_angle',
        'emv'
    ]].drop_duplicates()

    d["rotation"] = d["rotation"] * 2
    d.loc[d["emv"] > 160, "emv"] -= 360
    d["emv"] = d["emv"] - d["target_angle"]

    d = d.sort_values(by=['session', 'subject', 'trial'])

    return d


def fit_state_space_with_g_func_2_state():

    d = prep_for_fits()
    d = d.sort_values(by=['session', 'subject', 'trial', 'target_angle'])

    rot = d.loc[(d["session"] == 1) & (d["subject"] == 1),
                "rotation"].to_numpy()

    for i in d.session.unique():

        p_rec = np.empty((0, 8))

        x_obs = d.groupby(['session', 'target_angle',
                           'trial'])["emv"].mean().reset_index()
        x_obs = x_obs[x_obs['session'] == i]
        x_obs = x_obs[['emv', 'trial', 'target_angle']]
        x_obs = x_obs.pivot(index='trial',
                            columns='target_angle',
                            values='emv')
        x_obs = x_obs.values

        results = fit(x_obs, rot)
        p = results["x"]
        print(p)

        x_pred = simulate_state_space_with_g_func_2_state(p, rot)[0]
        fig, ax = plt.subplots(nrows=1, ncols=2)
        c = cm.rainbow(np.linspace(0, 1, 11))
        for k in range(11):
            ax[0].plot(x_obs[:, k], '.', color=c[k])
            ax[0].plot(x_pred[:, k], '-', color=c[k])
            ax[0].plot(rot, 'k')

        x = np.arange(0, 11, 1)
        y_obs = np.nanmean(x_obs[-65:-1, :], 0)
        y_pred = np.nanmean(x_pred[-65:-1, :], 0)
        ax[1].plot(x, y_obs)
        ax[1].plot(x, y_pred)
        plt.show()

        p_rec = np.append(p_rec, [p], axis=0)

        f_name_p = '../fits/fit_session_' + str(i) + '.txt'
        with open(f_name_p, 'w') as f:
            np.savetxt(f, p_rec, '%0.4f', '\n')


def fit_state_space_with_g_func_2_state_boot(n_boot_samp):

    d = prep_for_fits()

    rot = d.loc[(d["session"] == 1) & (d["subject"] == 1),
                "rotation"].to_numpy()

    for i in d.session.unique():

        p_rec = -1 * np.ones((1, 8))
        for b in range(n_boot_samp):
            print(i, b)

            subs = d['subject'].unique()
            boot_subs = np.random.choice(subs,
                                         size=subs.shape[0],
                                         replace=True)

            x_boot_rec = []
            for k in boot_subs:
                x_boot_rec.append(d[d['subject'] == k])
                x_boot = pd.concat(x_boot_rec)

            x_obs = x_boot.groupby(['session', 'target_angle',
                                    'trial'])["emv"].mean().reset_index()
            x_obs = x_obs[x_obs['session'] == i]
            x_obs = x_obs[['emv', 'target_angle', 'trial']]
            x_obs = x_obs.pivot(index='trial',
                                columns='target_angle',
                                values='emv')
            x_obs = x_obs.values
            results = fit(x_obs, rot)
            p_rec[0, :] = results["x"]

            f_name_p = '../fits/fit_session_' + str(i) + '_boot.txt'
            with open(f_name_p, 'a') as f:
                np.savetxt(f, p_rec, '%0.4f', ',')


def fit(x_obs, rot):
    # alpha_s = p[0]
    # beta_s = p[1]
    # g_sigma_s = p[2]
    # alpha_f = p[3]
    # beta_f = p[4]
    # g_sigma_f = p[5]
    # beta_s_2 = p[6]
    # beta_f_2 = p[7]
    constraints = LinearConstraint(A=[[1, 0, 0, -1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, -1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                   lb=[-1, -1, 0, 0, 0, 0, 0, 0],
                                   ub=[0, 0, 0, 0, 0, 0, 0, 0])

    args = (x_obs, rot)
    bounds = ((0, 1), (0, 1), (0, 150), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))

    results = differential_evolution(func=fit_obj_func_sse,
                                     bounds=bounds,
                                     constraints=constraints,
                                     args=args,
                                     maxiter=800,
                                     tol=1e-15,
                                     disp=False,
                                     polish=False,
                                     updating='deferred',
                                     workers=-1)
    return results


def inspect_results_boot():

    d = prep_for_fits()

    rot = d.loc[(d["session"] == 1) & (d["subject"] == 1),
                "rotation"].to_numpy()

    n_subs = d.subject.unique().shape[0]

    dd = d[d["phase"] == "generalization"][[
        'session', 'subject', 'target_angle', 'trial', 'emv'
    ]].groupby(['session', 'subject', 'target_angle',
                'trial']).mean().reset_index()

    # NOTE: model summary figure
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    c = cm.Set1(np.linspace(0, 1, 11))

    for i, s in enumerate(d.session.unique()):
        x_obs = d.groupby(['session', 'trial',
                           'target_angle'])["emv"].mean().reset_index()
        x_obs = x_obs[x_obs['session'] == s]
        x_obs = x_obs[['emv', 'target_angle', 'trial']]
        x_obs = x_obs.pivot(index='trial',
                            columns='target_angle',
                            values='emv')
        x_obs = x_obs.values

        p = np.loadtxt('../fits/fit_session_' + str(i + 1) + '.txt')

        x_pred = simulate_state_space_with_g_func_2_state(p, rot)[0]
        x_pred_s = simulate_state_space_with_g_func_2_state(p, rot)[1]
        x_pred_f = simulate_state_space_with_g_func_2_state(p, rot)[2]

        np.savetxt('../fits/x_pred_' + str(i + 1) + '_s.txt', x_pred_s)
        np.savetxt('../fits/x_pred_' + str(i + 1) + '_f.txt', x_pred_f)

        for k in range(11):
            ax[i, 0].plot(x_obs[:, k], '.', color=c[k], alpha=0.1)
            ax[i, 0].plot(x_pred_s[:, 5], ':', color=c[5])
            ax[i, 0].plot(x_pred_f[:, 5], '--', color=c[5])
            ax[i, 0].plot(x_pred[:, 5], '-', color=c[5])
            ax[i, 0].plot(rot, 'k')
            ax[i, 0].set_title('Session ' + str(i + 1), size=16)
            ax[i, 0].set_ylabel('Hand Angle', size=16)
            ax[i, 0].set_xlabel('Trial', size=16)
            ax[i, 0].set_ylim(-10, 65)
            ax[i, 0].set_xlim(0, 410)

        ddd = dd.groupby(['session', 'target_angle']).mean()
        ddd.reset_index(inplace=True)
        ddd = ddd[ddd['session'] == s][['target_angle', 'emv']]
        y_obs = ddd.emv.values

        ddd = dd.groupby(['session', 'target_angle']).std() / np.sqrt(n_subs)
        ddd.reset_index(inplace=True)
        ddd = ddd[ddd['session'] == s][['target_angle', 'emv']]
        y_obs_err = ddd.emv.values

        target_ind = np.array([3, 4, 5, 6, 7])
        x = np.arange(0, target_ind.shape[0], 1)

        y_pred = np.nanmean(x_pred[341:, target_ind], 0)
        y_pred_s = np.nanmean(x_pred_s[341:, target_ind], 0)
        y_pred_f = np.nanmean(x_pred_f[341:, target_ind], 0)
        ax[i, 1].errorbar(x, y_obs[target_ind], yerr=y_obs_err[target_ind])
        ax[i, 1].plot(x, y_pred, '.-')
        ax[i, 1].plot(x, y_pred_s, '.-', color='k', alpha=0.2)
        ax[i, 1].plot(x, y_pred_f, '--', color='k', alpha=0.2)
        ax[i, 1].set_ylim(-5, 60)
        ax[i, 1].set_title('Session ' + str(i + 1), size=16)
        ax[i, 1].set_ylabel('Hand Angle', size=16)
        ax[i, 1].set_xlabel('Relative Target Angle', size=16)
        ax[i, 1].set_xticks(x)
        ax[i, 1].set_xticklabels(['-60', '-30', '0', '30', '60'])

    axx = ax.flatten()
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, ax_ in enumerate(axx):
        ax_.text(-0.1,
                 1.1,
                 labels[i],
                 transform=ax_.transAxes,
                 size=20,
                 weight='bold',
                 va='top',
                 ha='right')

    plt.tight_layout()
    plt.savefig('../figures/fig_results.pdf')
    plt.close()

    # NOTE: model parameter distributions
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, s in enumerate(d.session.unique()):

        names = [
            'alpha_s', 'beta_s', 'g_s', 'alpha_f', 'beta_f', 'g_f', 'beta_s_2',
            'beta_f_2'
        ]

        d_boot = pd.read_csv('../fits/fit_session_' + str(s) + '_boot.txt',
                             names=names)

        d_boot = d_boot[[
            'alpha_s', 'alpha_f', 'beta_s', 'beta_f', 'g_s', 'g_f', 'beta_s_2',
            'beta_f_2'
        ]]

        d_boot['g_s'] = d_boot['g_s'] / 60.0
        d_boot['beta_s'] = 1 - d_boot['beta_s']
        d_boot['beta_f'] = 1 - d_boot['beta_f']

        p = np.loadtxt('../fits/fit_session_' + str(i + 1) + '.txt')
        p[1] = 1 - p[1]
        p[4] = 1 - p[4]
        p[2] = p[2] / 60.0

        # alpha_s = p[0]
        # beta_s = p[1]
        # g_sigma_s = p[2]
        # alpha_f = p[3]
        # beta_f = p[4]
        # g_sigma_f = p[5]
        # beta_s_2 = p[6]
        # beta_f_2 = p[7]
        p = np.array([p[0], p[3], p[1], p[4], p[2], p[5], p[6], p[7]])

        c = ['C0', 'C1', 'C2', 'C3']
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * 2
        ax.violinplot(d_boot.values, positions=x - 0.5 * i, showextrema=False)
        ax.boxplot(d_boot.values,
                   positions=x - 0.5 * i,
                   whis=[2.5, 97.5],
                   showfliers=False,
                   widths=0.4,
                   labels=None,
                   patch_artist=False)
        ax.plot(x - 0.5 * i, p, '.', color=c[i], label='Session ' + str(i + 1))
        ax.set_xticks(x)
        ax.set_xticklabels([
            r'$\boldsymbol{\alpha_{s}}$', r'$\boldsymbol{\alpha_{f}}$',
            r'$\boldsymbol{\beta_{s}}$', r'$\boldsymbol{\beta_{f}}$',
            r'$\boldsymbol{g_{s}}$', r'$\boldsymbol{g_{f}}$',
            r'$\boldsymbol{\gamma_{s}}$', r'$\boldsymbol{\gamma_{f}}$'
        ],
                           fontsize=16)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Parameter Value', size=16)

    plt.legend()
    plt.tight_layout()
    plt.savefig('../figures/fig_params.pdf')
    plt.close()


def inspect_boot_stats():
    names = [
        'alpha_s', 'beta_s', 'g_s', 'alpha_f', 'beta_f', 'g_f', 'beta_s_2',
        'beta_f_2'
    ]

    p0 = np.loadtxt('../fits/fit_group_0.txt')
    p1 = np.loadtxt('../fits/fit_group_1.txt')

    d0 = pd.read_csv('../fits/fit_group_0_boot.txt', names=names)
    d1 = pd.read_csv('../fits/fit_group_1_boot.txt', names=names)

    print(
        'alpha_s',
        str(
            bootstrap_t(p0[0], p1[0], d0.alpha_s.values, d1.alpha_s.values,
                        10000)))
    print(
        'beta_s',
        str(
            bootstrap_t(p0[1], p1[1], d0.beta_s.values, d1.beta_s.values,
                        10000)))
    print('g_s',
          str(bootstrap_t(p0[2], p1[2], d0.g_s.values, d1.g_s.values, 10000)))
    print(
        'alpha_f',
        str(
            bootstrap_t(p0[3], p1[3], d0.alpha_f.values, d1.alpha_f.values,
                        10000)))
    print(
        'beta_f',
        str(
            bootstrap_t(p0[4], p1[4], d0.beta_f.values, d1.beta_f.values,
                        10000)))
    print('g_f',
          str(bootstrap_t(p0[5], p1[5], d0.g_f.values, d1.g_f.values, 10000)))
    print(
        'beta_s_2',
        str(
            bootstrap_t(p0[6], p1[6], d0.beta_s_2.values, d1.beta_s_2.values,
                        10000)))
    print(
        'beta_f_2',
        str(
            bootstrap_t(p0[7], p1[7], d0.beta_f_2.values, d1.beta_f_2.values,
                        10000)))
