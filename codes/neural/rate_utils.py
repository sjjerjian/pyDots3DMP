# %% imports

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.ndimage import convolve1d
from scipy.signal import gaussian

from typing import Union
from neural.NeuralDataClasses import PseudoPop

# in all functions,
#    f_rates denotes a unit x trial x time numpy array of firing rates
#    fr_list denotes a list of f_rates arrays, one per interval/alignment


def concat_aligned_rates(fr_list, tvecs=None) -> tuple[Union[list, tuple], Union[list, tuple]]:
    """
    given multiple f_rates matrices stored in a list (e.g. different alignments),
    stack them.
    so instead of having to loop over alignments, we can apply functions on all firing rates at once
    this works for a list of lists (e.g. a list of multiple recording populations)
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


def build_pseudopop(popn_dfs, tr_tab, 
                    t_params: dict, smooth_params: dict = None,
                    return_averaged=True) -> PseudoPop:

    if smooth_params is None:
        params = {'type': 'boxcar', 'binsize': 0.05, 'width': 0.2, 'sigma': 0.05}

    # TODO this shouldn't be necessary
    align_ev = t_params['align_ev']
    trange = t_params['trange']
    binsize = t_params['binsize']

    # calculate firing rates on each trial, given PSTH parameters
    fr_list, unitlabels, conds_dfs, tvecs, _ = zip(*popn_dfs.apply(
        lambda x: x.get_firing_rates(align_ev=align_ev, trange=trange,
                                     binsize=binsize, sm_params=smooth_params,
                                     condlabels=tr_tab.columns)
        )
    )

    if return_averaged:
        if binsize == 0:
            rates_cat, len_intervals = concat_aligned_rates(fr_list)
        else:
            rates_cat, len_intervals = concat_aligned_rates(fr_list, tvecs)

        cond_frs, cond_groups = [], []
        for f_in, cond in zip(rates_cat, conds_dfs):

            # avg firing rate over time across units, each cond, per session
            f_out, _, cg = condition_averages(f_in, cond, cond_groups=tr_tab)
            cond_frs.append(f_out)
            cond_groups.append(cg)

        # resplit by len_intervals, for pseudopop creation
        fr_list = list(map(lambda f, x: np.split(f, x, axis=2)[:-1], cond_frs, len_intervals))

        # re-assign conds_dfs to unique conditions
        conds_dfs = cond_groups

    if 'other_ev' in t_params:
        for popn in popn_dfs:
            rel_event_times = popn.popn_rel_event_times(align=t_params['align_ev'], 
                                                        others=t_params['other_ev'])

    # conds_dfs = [df.assign(trialNum=np.arange(len(df))) for df in conds_dfs]

    # stack firing rates, along unit axis, with insertion on time axis according to t_idx
    num_units = np.array([x[0].shape[0] for x in fr_list])
    max_trs = max(list(map(len, list(conds_dfs))))

    stacked_frs = []

    # to stack all frs with time-resolutions preserved, make a single unique time vector (t_unq)
    # and insert each population fr matrix according to how its tvec lines up with t_unq
    # need to do this to handle variable start/end references, different sessions might have different lengths
    # e.g. motionOn - motionOff varies on each trial, and the limit across sessions will also vary
    # only do it if tvecs is specified, otherwise assume we are just using the interval averages

    t_unq, t_idx = [], []
    for j in range(len(fr_list[0])):

        u_pos = 0
        if binsize > 0:     #tvecs is not None  # or fr_list[0][0].ndim == 2

            if j==0:
                print("time vector provided, concatenating time-resolved firing rates into pseudo-population\n")

            concat_tvecs = [tvecs[i][j] for i in range(len(tvecs))]
            t_unq.append(np.unique(np.concatenate(concat_tvecs)))
            t_idx.append([np.ravel(np.where(np.isin(t, t_unq[j]))) for t in concat_tvecs])

            stacked_frs.append(np.full([num_units.sum(), max_trs, len(t_unq[j])], np.nan))
            for sess in range(len(fr_list)):
                stacked_frs[j][u_pos:u_pos+num_units[sess], 0:len(conds_dfs[sess]), t_idx[j][sess]] = fr_list[sess][j]
                u_pos = u_pos + num_units[sess]

        else:
            if j==0:
                print("no time provided, concatenating interval average rates into pseudo-population\n")

            stacked_frs.append(np.full([num_units.sum(), max_trs], np.nan))
            for sess in range(len(fr_list)):
                stacked_frs[j][u_pos:u_pos + num_units[sess], 0:len(conds_dfs[sess])] = np.squeeze(fr_list[sess][j])
                u_pos = u_pos + num_units[sess]

    u_idx = np.array([i for i, n in enumerate(num_units) for _ in range(n)])
    area = [p.area for fr, p in zip(fr_list, popn_dfs) for _ in range(fr[0].shape[0])] # TODO deal with area=None

    # stacked_conds = [conds_dfs[u] for u in u_idx]

    pseudo_pop = PseudoPop(
        subject=popn_dfs[0].subject,
        firing_rates=stacked_frs,
        timestamps=t_unq,
        psth_params=t_params,
        conds=conds_dfs,
        clus_group=np.hstack(unitlabels),
        area=area,
        unit_session=u_idx,
    )

    return pseudo_pop

