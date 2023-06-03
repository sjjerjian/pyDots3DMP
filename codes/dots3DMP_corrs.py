#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:26:29 2023

@author: stevenjerjian
"""

import numpy as np
import pdb

from scipy.stats import pearsonr, zscore
from scipy.special import comb
from itertools import combinations

from modules.dots3DMP_FRutils import condition_index

# %%  helper functions


def zscore_bygroup(arr, grp, axis=None):

    zscore_subarrs = []
    for group in np.unique(grp):

        sub_arr = np.compress(grp == group, arr, axis=axis)
        z_arr = zscore(sub_arr, axis=axis)
        zscore_subarrs.append(z_arr)

    return np.concatenate(zscore_subarrs, axis=axis)


def pearsonr_dropna(x, y):
    nidx = np.logical_or(np.isnan(x), np.isnan(x))
    corr, pvals = pearsonr(x[~nidx], y[~nidx])
    return corr, pvals

# %% correlation functions


# could combine these into one function that does both depending on
# how f_rates is provided (1 element or 2)


def corr_popn(f_rates, condlist, cond_groups, cond_columns, rtype=''):
    # pairwise correlations within a population (nChoose2 pairs)

    cg = cond_groups[cond_columns[:-1]].drop_duplicates()
    ic, nC, cg = condition_index(condlist[cond_columns[:-1]], cg)

    pair_corrs = []
    pair_pvals = []

    for c in range(nC):
        if np.sum(ic == c):
            # currently f_rates must be units x trials (no time/interval axis)
            # to be improved - so that we don't need a loop over third axis
            # outside the function (i.e. so it can also operate on cat rates)
            temp = f_rates[:, ic == c]

            # zscore over trials for each unit and heading separately
            if rtype == 'noise':
                hdgs = condlist.loc[ic == c, cond_columns[-1]].to_numpy()

                try:
                    temp = zscore_bygroup(temp, hdgs, axis=1)
                except ValueError:
                    pdb.set_trace()

            pairs = combinations(np.split(temp, temp.shape[0], axis=0), 2)

            try:
                corr, pval = zip(*[(pearsonr_dropna(pair[0][0], pair[1][0]))
                                   for pair in pairs])
            except ValueError:
                corr = [np.nan]*int(comb(temp.shape[0], 2))
                pval = [np.nan]*int(comb(temp.shape[0], 2))

        else:
            corr = [np.nan]*int(comb(f_rates.shape[0], 2))
            pval = [np.nan]*int(comb(f_rates.shape[0], 2))

        pair_corrs.append(np.array(corr))
        pair_pvals.append(np.array(pval))

    return pair_corrs, pair_pvals, cg


def corr_popn2(f_rates1, f_rates2, condlist, cond_groups, cond_columns,
               rtype=''):

    # pairwise correlations across two populations (nxm pairs)
    cg = cond_groups[cond_columns[:-1]].drop_duplicates()
    ic, nC, cg = condition_index(condlist[cond_columns[:-1]], cg)

    nUnits = [f_rates1.shape[0], f_rates2.shape[0]]
    pair_corrs = np.ndarray((nUnits[0], nUnits[1], nC))
    pair_pvals = np.ndarray((nUnits[0], nUnits[1], nC))

    for c in range(nC):
        if np.sum(ic == c):
            f1 = f_rates1[:, ic == c]
            f2 = f_rates2[:, ic == c]

            # zscore over trials for each unit and heading separately
            if rtype == 'noise':
                hdgs = condlist.loc[ic == c, cond_columns[-1]]
                f1 = zscore_bygroup(f1, hdgs, axis=1)
                f2 = zscore_bygroup(f2, hdgs, axis=1)

            for i in range(nUnits[0]):
                for j in range(nUnits[1]):
                    corr, pval = pearsonr(f1[i, :], f2[j, :])

                    pair_corrs[i, j, c] = corr
                    pair_pvals[i, j, c] = pval

    return pair_corrs, pair_pvals, cg
