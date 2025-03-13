import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pingouin as pg
from util_func_dbm import *
import statsmodels.api as sm
import patsy
from statsmodels.nonparametric.smoothers_lowess import lowess
from util_func_analysis import *

if __name__ == "__main__":

    d = load_data()

    if not os.path.exists("../dbm_fits/dbm_results.csv"):
        fit_dbm_models()
    else:
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")

    dd, ddd = get_best_model_class(dbm)

    make_fig_dbm(d, dd, ddd)

    make_fig_accuracy_per_block_by_model(d, dd, ddd)

    report_stats_learning_curve(d, dd, ddd)

    make_fig_switch_cost(d, dd, ddd)
