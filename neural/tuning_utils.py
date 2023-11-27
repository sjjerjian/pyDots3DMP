import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional
from scipy.optimize import curve_fit
from scipy.special import i0
from scipy.stats import (
    f_oneway, kruskal, ttest_rel, ttest_1samp, wilcoxon, vonmises
)

from neural.rate_utils import condition_index


# %% simple tuning statistics

def tuning_within(f_rates: np.ndarray, condlist: pd.DataFrame, cond_groups: Optional[pd.DataFrame] = None, 
                  cond_cols = None, tuning_col: str = 'heading', parametric: bool = True):
    """
    tuning within each time bin/interval (across tuning_col, with each cond_group in cond_groups)
    """
    
    stat_func = f_oneway if parametric else kruskal
    
    # TODO add a wrapper on condition index to handle this (or just add a third argument for cols...)
    if cond_groups is None:
        ic, nC, cond_groups = condition_index(condlist)
    
    if cond_cols is None:
        cond_cols = cond_groups.columns[~cond_groups.columns.str.contains('heading')]
        
    cg = cond_groups[cond_cols].drop_duplicates()
    ic, nC, cg = condition_index(condlist[cond_cols], cg)
    
    #f_stat and p_val results are units x conditions x time
    f_stat = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)
    p_val = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)

    for c in range(nC):
        if np.sum(ic == c):
            y = f_rates[:, ic == c, :]
            x = np.squeeze(condlist.loc[ic == c, tuning_col].to_numpy())

            y_grp = [y[:, x == g, :] for g in np.unique(x)]

            f, p = stat_func(*y_grp, axis=1)
            
            p[np.isnan(p)] = .99

            f_stat[:, c, :] = f
            p_val[:, c, :] = p

    return f_stat, p_val, cg


def tuning_across(f_rates: np.ndarray, condlist: pd.DataFrame, cond_groups: Optional[pd.DataFrame],
                  tuning_col: str = 'heading', 
                  bsln_t=0, abs_diff: bool = True, parametric: bool = True):
    """
    tuning at each time/interval, relative to a baseline time, across conditions
    """
    
    ic, nC, cg = condition_index(condlist, cond_groups)

    # result is units x conditions x time
    stats = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)
    p_val = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)

    for c in range(nC):
        if np.sum(ic == c):

            y0 = f_rates[:, ic == c, bsln_t]
            y = f_rates[:, ic == c, :]

            if y0.ndim == 3:
                y0 = np.mean(y0, axis=2) # average over bsln_t time period

            for t in range(y.shape[2]):
                yt = y[:, :, t]
                if parametric:
                    if abs_diff:
                        stat, p = ttest_1samp(np.abs(yt-y0), 0, axis=1)
                    else:
                        stat, p = ttest_rel(y0, yt, axis=1)
                else: 
                    if abs_diff:
                        stat, p = wilcoxon(np.abs(yt - y0), axis=1)
                    else:
                        stat, p = wilcoxon(yt, y0, axis=1)

                p[np.isnan(p)] = .99

                stats[:, c, t] = stat
                p_val[:, c, t] = p


    return stats, p_val, cg


# %% von Mises tuning curve fits

# TODO decorate these to allow variable function inputs


def tuning_vonMises(x, a, k, theta, b):
    return a * np.exp(k * np.cos(np.deg2rad(x - theta))) / \
        (2 * np.pi * i0(k)) + b
    # return a * vonmises.pdf(x, k, loc=theta, scale=1) + b


# could just merge this into the fit_predict function
# def fit_vonMises(y, x):
#     p0 = [np.max(y) - np.min(y), 1.5, x[np.argmax(y)], np.min(y)]
#     popt, pcov = curve_fit(tuning_vonMises, x, y, p0=p0)
#     perr = np.sqrt(np.diag(pcov))  # 1SD error on parameters
#     return popt, perr


def plot_vonMises_fit(func):
    def plot_wrapper(y, x, x_pred):
        y_pred = func(y, x, x_pred)
        plt.plot(x, y, label='y')
        plt.plot(x_pred, y_pred, label='y_pred')
        plt.legend()
        plt.show()
        return y_pred
    return plot_wrapper


# @plot_vonMises_fit
def fit_predict_vonMises(y, x, x_pred, k=1.5):
    # popt, perr = fit_vonMises(y, x)

    # TODO bug fixes needed
    # can this work on 2-D y array
    if y.any():
        p0 = np.array([np.max(y) - np.min(y), k, x[np.argmax(y)], np.min(y)])
    else:
        p0 = np.array([1, k, 0, 0])

    try:
        popt, pcov = curve_fit(tuning_vonMises, x, y, p0=p0)
    except RuntimeError as re:
        print(re)
        popt = np.full(p0.shape, np.nan)
        perr = np.full(p0.shape, np.nan)
        y_pred = np.full(x_pred.shape, np.nan)
    except ValueError as ve:
        print(ve)
        popt = np.full(p0.shape, np.nan)
        perr = np.full(p0.shape, np.nan)
        y_pred = np.full(x_pred.shape, np.nan)
    except TypeError:
        pdb.set_trace()
    else:
        perr = np.sqrt(np.diag(pcov))  # 1SD error on parameters
        y_pred = tuning_vonMises(x_pred, *popt)

    return y_pred, popt, p0, perr


# %% preferred heading calculations

def delta_pref_hdg(func):
    def wrapper(y, x, axis=1):
        pref_hdg, pref_dir = func(y, x, axis)
        nUnits = pref_hdg.shape[0]
        pref_hdg_diffs = np.zeros((nUnits, nUnits))
        for i in range(nUnits):
            for j in range(nUnits):
                pref_hdg_diffs[i, j] = abs(pref_hdg[i] - pref_hdg[j])
        return pref_hdg, pref_dir, pref_hdg_diffs
    return wrapper


@delta_pref_hdg
def tuning_basic(y, x, axis=1):

    # assume y is units x conditions
    # x is conditions 1-D

    yR = y[:, x > 0]
    yL = y[:, x < 0]

    pref_hdg = x[np.argmax(np.abs(y - np.mean(y, axis=axis,
                                              keepdims=True)),
                           axis=axis)]

    #pref_mag = (np.mean(yR, axis=axis) - np.mean(yL, axis=axis))
    #pref_dir = np.sign(pref_mag)

    pref_dir = (np.mean(yR, axis=axis) > np.mean(yL, axis=axis)).astype(int)
    
    return pref_hdg, pref_dir

