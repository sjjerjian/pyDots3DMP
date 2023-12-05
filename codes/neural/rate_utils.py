# %% imports

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.ndimage import convolve1d
from scipy.signal import gaussian

from typing import Union, Optional

# in all functions,
#    f_rates denotes a unit x trial x time numpy array of firing rates
#    fr_list denotes a list of f_rates arrays, one per interval/alignment


def concat_aligned_rates(fr_list: list[np.ndarray], tvecs: Optional[np.ndarray]=None,
                         insert_blank: bool = False) -> tuple[Union[list, tuple], Union[list, tuple]]:
    """
    given multiple f_rates matrices stored in a list (e.g. different alignments),
    stack them.
    so instead of having to loop over alignments, we can apply functions on all firing rates at once
    this works for a list of lists (e.g. a list of multiple recording populations
    """

    if tvecs is not None:
        rates_cat, tvecs_cat, len_intervals = [], [], []

        for f, t in zip(fr_list, tvecs):
            rates_s, tvecs_s, len_s = concat_aligned_rates_single(f, tvecs=t, insert_blank=insert_blank)
            rates_cat.append(rates_s)
            tvecs_cat.append(tvecs_s)
            len_intervals.append(len_s)

    else:
        # each 'interval' is length 1 if binsize was set to 0,
        # so can simply 'dstack' each sessions rates
        # insert_blank argument is ignored

        rates_cat = list(map(np.dstack, fr_list))
        tvecs_cat = None
        len_intervals = None

    return rates_cat, tvecs_cat, len_intervals


def concat_aligned_rates_single(frs: list[np.ndarray], tvecs: Optional[np.ndarray]=None,
                                insert_blank: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if tvecs is not None:
        rates_cat = np.concatenate(frs, axis=2)
        tvecs_cat = np.concatenate(tvecs)

        # concatenate all different alignments, but store the lens for later splits
        len_intervals = np.array(list(map(lambda x: x.size, tvecs)), dtype='int').cumsum()

        if insert_blank:
            # insert NaN column at points where alignments are concatenated
            # add (i-1) to move along as we are updating the array within the loop!
            for i, intvl in enumerate(len_intervals[:-1]):
                rates_cat = np.insert(rates_cat, intvl+i, np.nan, axis=2)
                tvecs_cat = np.insert(tvecs_cat, intvl+i, np.nan)

    else:
        # each 'interval' is length 1, if binsize was set to 0
        rates_cat = np.dstack(frs)
        tvecs_cat = None
        len_intervals = None

    return rates_cat, tvecs_cat, len_intervals


def mask_low_firing(f_rates: np.ndarray, minfr: int = 0) -> np.ndarray:

    # TODO find a way to not exclude units not recorded in all conditions!
    mean_fr = np.squeeze(np.nanmean(f_rates, axis=1)) # across conditions
    lowfr_units = np.logical_or(np.isnan(mean_fr), mean_fr <= minfr)
    lowfr_units = mean_fr <= minfr

    return lowfr_units


# %% trial condition helpers

def condition_index(condlist: pd.DataFrame, cond_groups: Optional[pd.DataFrame]=None) -> tuple[np.ndarray, int, pd.DataFrame]:
    """
    given a single trial conditions list, and a unique set of conditions,
    return the trial index for each condition
    """
    if cond_groups is None:
        # cond_groups not specified, use all unique trial types
        cond_groups, ic = np.unique(condlist.to_numpy('float64'), axis=0, return_inverse=True)
        cond_groups = pd.DataFrame(cond_groups, columns=condlist.columns)

    else:
        # cond_groups user-specified
        assert isinstance(cond_groups, pd.DataFrame)
        cond_groups = cond_groups.loc[:, condlist.columns]

        ic = np.full(condlist.shape[0], fill_value=-1, dtype=int)
        for i, cond in enumerate(cond_groups.values):
            ic[(condlist == cond).all(axis=1)] = i

    nC = len(cond_groups.index)
    return ic, nC, cond_groups


def condition_averages(f_rates, condlist, cond_groups: Optional[pd.DataFrame]=None, return_sem: bool=True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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
    cond_err = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)

    # loop over conditions, calculate firing mean and SE
    for c in range(nC):
        if np.sum(ic == c):
            cond_fr[:, c, :] = np.mean(f_rates[:, ic == c, :], axis=1)
            cond_err[:, c, :] = np.std(f_rates[:, ic == c, :], axis=1)

            if return_sem:
                cond_err[:, c, :] /= np.sqrt(np.sum(ic == c))

    # to avoid divide by zero errors?
    cond_err[cond_err==0] = 1

    return cond_fr, cond_err, cond_groups


def demean_conditions(f_rates, condlist, cond_groups, standardize=True):
    """Demean/standardize trial frs each timepoint for each unit and condition separately

    Args:
        f_rates (_type_): _description_
        condlist (_type_): _description_
        cond_groups (_type_): _description_
        standardize (bool, optional): _description_. Defaults to True.
    """
    ic, nC, cond_groups = condition_index(condlist, cond_groups)
    cond_fr, cond_sd, _ = condition_averages(f_rates, condlist, cond_groups, return_sem=False)

    f_rates_demeaned = np.full_like(f_rates, np.nan)

    for c in range(len(cond_groups)):

        f_rates_demeaned[:, ic==c, :] = f_rates[:, ic==c, :] - np.expand_dims(cond_fr[:, c, :], axis=1)

        if standardize:

            f_rates_demeaned[:, ic==c, :] /= np.expand_dims(cond_sd[:, c, :], axis=1)

    return f_rates_demeaned, cond_fr, cond_groups


def pref_hdg_dir(f_rates: np.ndarray, condlist: pd.DataFrame, cond_groups: Optional[pd.DataFrame]=None,
                 cond_cols=None, method: str = 'peak') -> np.ndarray:

    # f_rates should be raw, or fit
    # for each row in cond_groups, calulate preferred heading (0 (left), or 1 (right) for each condition in cond_groups)

    if cond_groups is None:
        ic, nC, cond_groups = condition_index(condlist)

    if cond_cols is None:
        cond_cols = cond_groups.columns[~cond_groups.columns.str.contains('heading')]

    cg = cond_groups[cond_cols].drop_duplicates()
    ic, nC, cg = condition_index(condlist[cond_cols], cg)

    if f_rates.ndim == 2:
        f_rates = f_rates[:, :, np.newaxis]

    pref_dir = np.full((f_rates.shape[0], nC, f_rates.shape[2]), fill_value=np.nan)
    pref_fr = np.full((f_rates.shape[0], nC, f_rates.shape[2]), fill_value=np.nan)

    for c in range(nC):

        fr_c = f_rates[:, ic==c, ...]
        fr_c[np.isnan(fr_c)] = 0

        # no trials of that condition
        if fr_c.sum() == 0:
            continue

        hdg_c = condlist.loc[ic==c, 'heading'].values

        r_minus_l = np.nansum(fr_c[:, hdg_c > 0, ...], axis=1) - np.nansum(fr_c[:, hdg_c < 0, ...], axis=1)
        r_plus_l = np.nansum(fr_c[:, hdg_c > 0, ...], axis=1) + np.nansum(fr_c[:, hdg_c < 0, ...], axis=1)

        if method == 'peak':
            pref_dir[:, c, :] = np.sign(hdg_c[np.argmax(fr_c, axis=1)])

        elif method == 'sum':
            # r_minus_l = np.sum(fr_c[:, hdg_c > 0, ...], axis=1) - np.sum(fr_c[:, hdg_c < 0, ...], axis=1)
            pref_dir[:, c, :] = np.sign(r_minus_l)


        elif method == 'trapz':
            r_minus_l = np.trapz(fr_c[:, hdg_c >= 0, ...], x=hdg_c[hdg_c >= 0]) - \
                np.trapz(fr_c[:, hdg_c <= 0, ...], x=hdg_c[hdg_c <= 0])
            pref_dir[:, c, :] = np.sign(r_minus_l)

        pref_fr[:, c, :] =  r_minus_l

    # set zeros to NaN, and -1 (i.e. left) to 0
    pref_dir[pref_dir==0] = np.nan
    pref_dir[pref_dir==-1] = 0

    return np.squeeze(pref_dir), np.squeeze(pref_fr)




