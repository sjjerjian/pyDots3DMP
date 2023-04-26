#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 00:08:10 2023

@author: stevenjerjian
"""

# compute tuning curves and signal correlations
# tuning curve can be a property of neuron class eventually, a an aside

import pdb
import numpy as np
import matplotlib.pyplot as plt

# from scipy.stats import vonmises
from scipy.optimize import curve_fit
from scipy.special import i0
from scipy.stats import f_oneway, kruskal

from dots3DMP_FRutils import condition_index

# %% von Mises

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

    pref_dir = (np.mean(yR, axis=axis) >
                np.mean(yL, axis=axis)).astype(int) + 1
    
    return pref_hdg, pref_dir


def tuning_sig(f_rates, condlist, cond_groups, cond_columns):

    cg = cond_groups[cond_columns[:-1]].drop_duplicates()
    ic, nC, cg = condition_index(condlist[cond_columns[:-1]], cg)

    f_stat = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)
    p_val = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)

    for c in range(nC):
        if np.sum(ic == c):
            y = f_rates[:, ic == c, :]
            x = np.squeeze(condlist.loc[ic == c, cond_columns[-1]].to_numpy())

            y_grp = [y[:, x == g, :] for g in np.unique(x)]
            f, p = f_oneway(*y_grp, axis=1)
            #f, p = kruskal(*y_grp, axis=1)

            f_stat[:, c, :] = f
            p_val[:, c, :] = p

    return f_stat, p_val, cg
