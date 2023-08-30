# %% imports

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.ndimage import convolve1d
from scipy.signal import gaussian

from typing import Union

# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(context="notebook", style="ticks", rc=custom_params)

# in all functions,
#    f_rates denotes a unit x trial x time numpy array of firing rates
#    fr_list denotes a list of f_rates arrays, one per interval/alignment


def concat_aligned_rates(fr_list, tvecs=None) -> tuple[Union[list, tuple], Union[list, tuple]]:
    """
    given multiple f_rates matrices stored in a list (e.g. different alignments),
    stack them.
    so instead of having to loop over alignments, we can apply functions on all firing rates at once

    """

    if tvecs is not None:
        # concatenate all different alignments...
        # but store the lens for later splits
        len_intervals = [np.asarray(list(map(lambda x: x.size, t)),
                         dtype='int').cumsum() for t in tvecs]
        rates_cat = list(map(lambda x: np.concatenate(x, axis=2), fr_list))
    else:
        # each 'interval' is length 1 if binsize was set to 0
        len_intervals = [np.ones(len(r), dtype=int).cumsum() for r in fr_list]
        rates_cat = list(map(np.dstack, fr_list))

    return rates_cat, len_intervals

def mask_low_firing(f_rates: np.ndarray, minfr=0, dim=1) -> np.ndarray:

    # TODO find a way to not exclude units not recorded in all conditions!
    mean_fr = np.squeeze(np.nanmean(f_rates, axis=dim))
    lowfr_units = np.logical_or(np.isnan(mean_fr), mean_fr <= minfr)
    lowfr_units = mean_fr <= minfr

    return lowfr_units


# %% trial condition helpers

def condition_index(condlist: pd.DataFrame, cond_groups=None) -> tuple[np.ndarray, int, pd.DataFrame]:
    """
    given a single trial conditions list, and a specified unique set of conditions,
    return the trial index for each condition
    """
    if cond_groups is None:
        cond_groups, ic = np.unique(condlist.to_numpy('float64'), axis=0,
                                    return_inverse=True)
        cond_groups = pd.DataFrame(cond_groups, columns=condlist.columns)

    else:
        # cond_groups user-specified
        assert isinstance(cond_groups, pd.DataFrame)
        cond_groups = cond_groups.loc[:, condlist.columns]

        # fill with nan?
        ic = np.full(condlist.shape[0], fill_value=-1, dtype=int)
        for i, cond in enumerate(cond_groups.values):
            ic[(condlist == cond).all(axis=1)] = i

    nC = len(cond_groups.index)
    return ic, nC, cond_groups


def condition_averages_ds(ds, *args):
    # xarray groupby doesn't support multiple columns, this is low priority
    ...


def condition_averages(f_rates, condlist, cond_groups=None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    calculate condition-averaged firing rates
    f_rates: single trial firing rates [units x trials x time/interval]
    condlist: single trial conditions dataframe
    cond_groups: unique conditions to calculate averages for (if None, then just take uniques from condlist)
    """

    ic, nC, cond_groups = condition_index(condlist, cond_groups)

    # condition-averaged arrays will have same number of units and time bins,
    # so match dim 0 and 2, but set dim 1 to number of conditions (nC)
    cond_fr = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)
    cond_sem = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)

    # loop over conditions, calculate firing mean and SE
    for c in range(nC):
        if np.sum(ic == c):
            cond_fr[:, c, :] = np.mean(f_rates[:, ic == c, :], axis=1)
            cond_sem[:, c, :] = np.std(f_rates[:, ic == c, :],
                                       axis=1) / np.sqrt(np.sum(ic == c))

    return cond_fr, cond_sem, cond_groups


