import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import multivariate_normal
from scipy import signal
from scipy.stats import norm
from scipy.stats import binom
from scipy.optimize import differential_evolution, LinearConstraint
from scipy import stats
import pingouin as pg


def fit_dbm(d, model_func, side, k, n, model_name):
    fit_args = {
        "obj_func": None,
        "bounds": None,
        "disp": False,
        "maxiter": 3000,
        "popsize": 20,
        "mutation": 0.7,
        "recombination": 0.5,
        "tol": 1e-3,
        "polish": False,
        "updating": "deferred",
        "workers": -1,
    }

    obj_func = fit_args["obj_func"]
    bounds = fit_args["bounds"]
    maxiter = fit_args["maxiter"]
    disp = fit_args["disp"]
    tol = fit_args["tol"]
    polish = fit_args["polish"]
    updating = fit_args["updating"]
    workers = fit_args["workers"]
    popsize = fit_args["popsize"]
    mutation = fit_args["mutation"]
    recombination = fit_args["recombination"]

    cnd = d["condition"]
    sub = d["subject"]
    cue = d["sub_task"]

    drec = []
    for m, mod in enumerate(model_func):
        dd = d[(d["subject"] == sub) & (d["condition"] == cnd) &
               (d["sub_task"] == cue)][["cat", "x", "y", "resp"]]

        cat = dd.cat.to_numpy()
        x = dd.x.to_numpy()
        y = dd.y.to_numpy()
        resp = dd.resp.to_numpy()

        # nll funcs expect resp to be [0, 1]
        n_zero = np.sum(resp == 0)
        n_one = np.sum(resp == 1)
        n_two = np.sum(resp == 2)
        n_three = np.sum(resp == 3)

        if np.argmax([n_zero, n_one, n_two, n_three]) > 1:
            resp = resp - 2

        # rescale x and y to be [0, 100]
        range_x = np.max(x) - np.min(x)
        x = ((x - np.min(x)) / range_x) * 100
        range_y = np.max(y) - np.min(y)
        y = ((y - np.min(y)) / range_y) * 100

        # compute glc bnds
        yub = np.max(y) + 0.1 * range_y
        ylb = np.min(y) - 0.1 * range_y
        bub = 2 * np.max([yub, -ylb])
        blb = -bub
        nlb = 0.001
        nub = np.max([range_x, range_y]) / 2

        if "unix" in model_name[m]:
            bnd = ((0, 100), (nlb, nub))
        elif "uniy" in model_name[m]:
            bnd = ((0, 100), (nlb, nub))
        elif "glc" in model_name[m]:
            bnd = ((-1, 1), (blb, bub), (nlb, nub))
        elif "gcc" in model_name[m]:
            bnd = ((0, 100), (0, 100), (nlb, nub))

        z_limit = 3

        args = (z_limit, cat, x, y, resp, side[m])

        results = differential_evolution(
            func=mod,
            bounds=bnd,
            args=args,
            disp=disp,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=polish,
            updating=updating,
            workers=workers,
        )

        tmp = np.concatenate((results["x"], [results["fun"]]))
        tmp = np.reshape(tmp, (tmp.shape[0], 1))

        # a1*x + a2*y + b = 0
        # y = -(a1*x + b) / a2
        a1 = results['x'][0]
        a2 = np.sqrt(1 - a1**2)
        b = results['x'][1]

        print(d[["condition", "subject", "sub_task"]].iloc[0])
        print(model_name[m], results["x"], results["fun"])
        print(a1, a2, b)
        print(np.unique(resp))

#        fig, ax = plt.subplots(1, 1, squeeze=False)
#        ax[0, 0].scatter(x, y, c=resp)
#        ax[0, 0].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], '--k')
#        ax[0, 0].set_xlim(-5, 105)
#        ax[0, 0].set_ylim(-5, 105)
#        plt.show()

        tmp = pd.DataFrame(results["x"])
        tmp.columns = ["p"]
        tmp["nll"] = results["fun"]
        tmp["bic"] = k[m] * np.log(n) + 2 * results["fun"]
        # tmp['aic'] = k[m] * 2 + 2 * results['fun']
        tmp["model"] = model_name[m]
        drec.append(tmp)

    drec = pd.concat(drec)
    return drec


def nll_unix(params, *args):
    """
    - returns the negative loglikelihood of the unidimensional X bound fit
    - params format:  [bias noise] (so x=bias is boundary)
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    xc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    zscoresX = (x - xc) / noise
    zscoresX = np.clip(zscoresX, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresX, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresX, 0.0, 1.0)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + sum(log_B_probs))

    return nll


def nll_uniy(params, *args):
    """
    - returns the negative loglikelihood of the unidimensional Y bound fit
    - params format:  [bias noise] (so y=bias is boundary)
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    yc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    zscoresY = (y - yc) / noise
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresY, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresY, 0.0, 1.0)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + sum(log_B_probs))

    return nll


def nll_glc(params, *args):
    """
    - returns the negative loglikelihood of the GLC
    - params format: [a1 b noise]
    -- a1*x+a2*y+b=0 is the linear bound
    -- assumes without loss of generality that:
    --- a2=sqrt(1-a1^2)
    --- a2 >= 0
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    a1 = params[0]
    a2 = np.sqrt(1 - params[0]**2)
    b = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    z_coefs = np.array([[a1, a2, b]]).T / params[2]
    data_info = np.array([x, y, np.ones(np.shape(x))]).T
    zscores = np.dot(data_info, z_coefs)
    zscores = np.clip(zscores, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscores)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscores)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + np.sum(log_B_probs))

    return nll


def nll_gcc_eq(params, *args):
    """
    returns the negative loglikelihood of the 2d data for the General
    Conjunctive Classifier with equal variance in the two dimensions.

    Parameters:
    params format: [biasX biasY noise] (so x = biasX and
    y = biasY make boundary)
    data row format:  [subject_response x y correct_response]
    z_limit is the z-score value beyond which one should truncate
    """

    xc = params[0]
    yc = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    if side == 0:
        zscoresX = (x - xc) / noise
        zscoresY = (y - yc) / noise
    if side == 1:
        zscoresX = (xc - x) / noise
        zscoresY = (y - yc) / noise
    if side == 2:
        zscoresX = (x - xc) / noise
        zscoresY = (yc - y) / noise
    else:
        zscoresX = (xc - x) / noise
        zscoresY = (yc - y) / noise

    zscoresX = np.clip(zscoresX, -z_limit, z_limit)
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    pXB = norm.cdf(zscoresX)
    pYB = norm.cdf(zscoresY)

    prB = pXB * pYB
    prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + np.sum(log_B_probs))

    return nll


def val_gcc_eq(params, *args):
    """
    Generates model responses for 2d data for the General Conjunctive
    Classifier with equal variance in the two dimensions.

    Parameters:
    params format: [biasX biasY noise] (so x = biasX and
    y = biasY make boundary)
    data row format:  [subject_response x y correct_response]
    z_limit is the z-score value beyond which one should truncate
    """

    xc = params[0]
    yc = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    if side == 0:
        zscoresX = (x - xc) / noise
        zscoresY = (y - yc) / noise
    if side == 1:
        zscoresX = (xc - x) / noise
        zscoresY = (y - yc) / noise
    if side == 2:
        zscoresX = (x - xc) / noise
        zscoresY = (yc - y) / noise
    else:
        zscoresX = (xc - x) / noise
        zscoresY = (yc - y) / noise

    zscoresX = np.clip(zscoresX, -z_limit, z_limit)
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    pXB = norm.cdf(zscoresX)
    pYB = norm.cdf(zscoresY)

    prB = pXB * pYB
    prA = 1 - prB

    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_glc(params, *args):
    """
    Generates model responses for 2d data in the GLC.
    - params format: [a1 b noise]
    -- a1*x+a2*y+b=0 is the linear bound
    -- assumes without loss of generality that:
    --- a2=sqrt(1-a1^2)
    --- a2 >= 0
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    a1 = params[0]
    a2 = np.sqrt(1 - params[0]**2)
    b = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    z_coefs = np.array([[a1, a2, b]]).T / params[2]
    data_info = np.array([x, y, np.ones(np.shape(x))]).T
    zscores = np.dot(data_info, z_coefs)
    zscores = np.clip(zscores, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscores)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscores)
        prA = 1 - prB

    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp
