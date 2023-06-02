#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:19:37 2023

@author: stevenjerjian
"""

# generate psth over trials(fixed alignements, or time-warping)
# average psth over conditions
# fano factor calculations
# multiple alignments, use map/apply? test in main

# %% imports

import numpy as np
import pandas as pd
import xarray as xr

from scipy.ndimage import convolve1d
from scipy.signal import gaussian

# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(context="notebook", style="ticks", rc=custom_params)

# in all functions,
#    f_rates denotes a unit x trial x time numpy array of firing rates
#    fr_list denotes a list of f_rates arrays, one per interval/alignment

# %% per-trial spike counts/firing rates


def trial_psth(spiketimes, align, trange,
               binsize=0.05, sm_params={},
               all_trials=False, normalize=True):
    """
    Parameters
    ----------
    spiketimes : numpy array
        1-D numpy array, containing the time of each spike for one unit.
    align : numpy array
        the times of the desired event(s) spikes should be aligned to,
        1 row per trial. Units should match spiketimes
        if 1-D, tstart and tend will be relative to the same event
        if 2-D, tstart will be relative to the first event, tend to the second
    trange : TYPE
        2-length 1-D numpy array, specifying start and end time
        relative to align_ev columns (again, units should be consistent)
    binsize : FLOAT, optional
        binsize for spike count histogram. The default is 0.05 (seconds).
    sm_params : DICT, optional
        DESCRIPTION. The default is {}.
    all_trials : BOOLEAN, optional
        DESCRIPTION. The default is False.
    normalize : BOOLEAN, optional
        normalize counts by time to obtain rate. The default is True.

    Returns
    -------
    fr_out : numpy array
        if binsize>0, ntrials x numbins array containing spike count or rate
        otherwise, 1-D array of total count or average rate on each trial
    x : numpy array
        if binsize>0, 1-D array containing the mid-time of each histogram bin
        otherwise, 1-D array of total time duration of each trial interval
    spktimes_aligned : list
        list of numpy arrays, containing individual spike times on each trial,
        relative to alignment event. Useful for plotting spike rasters

    """

    nTr = align.shape[0]

    if nTr == 0:
        return
    else:
        if align.ndim == 2:
            align = np.sort(align, axis=1)
            ev_order = np.argsort(align, axis=1)
            # TODO assertion here that ev_order is consistent on every trial?
            which_ev = ev_order[0, 0]  # align to this event
        else:  # only one event provided, tile it to standardize for later
            align = np.expand_dims(align, axis=1)
            align = np.tile(align, (1, 2))
            which_ev = 0

        nantrs = np.any(np.isnan(align), axis=1)

        if np.sum(nantrs) > 0:
            print(f'Dropping {np.sum(nantrs)} trials with missing event (NaN)')

        align = align[~nantrs, :]

        nTr = align.shape[0]  # recalculate after bad trs removed

        tr_starts = align[:, 0] + trange[0]
        tr_ends = align[:, 1] + trange[1]
        durs = tr_ends - tr_starts

        # compute 'corrected' tStart and tEnd based on align_ev input
        if which_ev == 1:
            tstarts_new = tr_starts - align[:, 1]
            tstart_new = np.min(tstarts_new)
            tend_new = trange[1]
            tends_new = tend_new.repeat(tstarts_new.shape[0])

        elif which_ev == 0:
            tends_new = tr_ends - align[:, 0]
            tend_new = np.max(tends_new)
            tstart_new = trange[0]
            tstarts_new = tstart_new.repeat(tends_new.shape[0])

        if binsize > 0:

            if trange[0] < 0 and trange[1] > 0:
                # ensure that time '0' is in between two bins exactly
                x0 = np.arange(0, tstart_new-binsize-1e-3, -binsize)
                x1 = np.arange(0, tend_new+binsize+1e-3, binsize)
                x = np.hstack((x0[::-1, ], x1[1:, ]))
            else:
                x = np.arange(tstart_new, tend_new+binsize, binsize)

            fr_out = np.full([nTr, x.shape[0]-1], np.nan)
        else:
            fr_out = np.full(nTr, np.nan)
            x = durs

        spktimes_aligned = []
        if spiketimes.any():
            itr_start, itr_end = 0, nTr

            if not all_trials:
                itr_start = np.argmin(np.abs(tr_starts - spiketimes[0]))
                itr_end = np.argmin(np.abs(tr_ends - spiketimes[-1]))

            for itr in range(0, itr_start):
                spktimes_aligned.append(np.empty(1))

            for itr in range(itr_start, itr_end+1):
                spk_inds = np.logical_and(spiketimes >= tr_starts[itr],
                                          spiketimes <= tr_ends[itr])

                if binsize == 0:
                    fr_out[itr] = np.sum(spk_inds)

                else:
                    inds_t = spiketimes[spk_inds] - align[itr, which_ev]
                    fr_out[itr, :], _ = np.histogram(inds_t, x)

                    spktimes_aligned.append(inds_t)

                    # set nans outside the range of align/trange for each trial
                    if which_ev == 0:
                        end_pos = np.argmin(abs(x - tends_new[itr]))
                        fr_out[itr, end_pos:] = np.nan
                    elif which_ev == 1:
                        start_pos = np.argmin(abs(x - tstarts_new[itr]))
                        fr_out[itr, :start_pos] = np.nan

            for itr in range(itr_end+1, nTr):
                spktimes_aligned.append(np.empty(1))

            if binsize > 0:
                x = x[:-1] + np.diff(x)/2  # shift x values to bin centers

                if sm_params:
                    fr_out = smooth_counts(fr_out, params=sm_params)

                if normalize:
                    fr_out /= binsize

            elif binsize == 0:
                if normalize:
                    fr_out /= x

        return fr_out, x, spktimes_aligned


def smooth_counts(raw_fr, params={'type': 'boxcar', 'binsize': 0.02,
                                  'width': 0.2, 'sigma': 0.05}):

    N = int(np.ceil(params['width'] / params['binsize']))  # width, in bins

    if params['type'] == 'boxcar':

        win = np.ones(N) / N

    elif params['type'] == 'gaussian':

        alpha = (N - 1) / (2 * (params['sigma'] / params['binsize']))
        win = gaussian(N, std=alpha)
        # win /= np.sum(win)  # win is already normalized to win in scipy

        # smoothed_fr = gaussian_filter1d(raw_fr, sigma=alpha)

    elif params['type'] == 'CHG':  # causal half-gaussian

        alpha = (N - 1) / (2 * (params['sigma'] / params['binsize']))
        win = gaussian(N, std=alpha)
        win[:(N//2)-1] = 0
        win /= np.sum(win)  # renormalize here

    smoothed_fr = convolve1d(raw_fr, win, axis=0, mode='nearest')
    return smoothed_fr


def concat_aligned_rates(fr_list, tvecs=None):

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


def lowfr_units(f_rates, minfr=0):

    mean_fr = np.squeeze(np.mean(f_rates, axis=1))
    lowfr_units = np.logical_or(np.isnan(mean_fr), mean_fr <= minfr)

    return lowfr_units


# %% trial condition helpers

def condition_index(condlist, cond_groups=None):

    assert isinstance(condlist, pd.DataFrame)

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
    # xarray groupby functions seem a bit clunky, this is low priority
    ...


def condition_averages(f_rates, condlist, cond_groups=None):

    ic, nC, cond_groups = condition_index(condlist, cond_groups)

    # condition-averaged arrays will have same number of units and time bins,
    # so match dim 0 and 2
    cond_fr = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)
    cond_sem = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)

    for c in range(nC):
        if np.sum(ic == c):
            cond_fr[:, c, :] = np.mean(f_rates[:, ic == c, :], axis=1)
            cond_sem[:, c, :] = np.std(f_rates[:, ic == c, :],
                                       axis=1) / np.sqrt(np.sum(ic == c))

    return cond_fr, cond_sem, cond_groups


    # if type == 'avg':
    #     sns.lineplot(data=df,
    #                  x='heading', y='spike counts',
    #                  estimator='mean', errorbar='se', err_style='bars',
    #                  hue=df[condlabels[:-1]].apply(tuple, axis=1))
    # elif type == 'time':
    #     g = sns.relplot(data=df,
    #                     x='time', y='firing_rate',
    #                     hue='heading', row='modality', col='align',
    #                     palette=palette,
    #                     facet_kws=dict(sharex=False),
    #                     kind='line', aspect=.75)

# %% Extract firing rates for population

# this can be deprecated since it's now defined as a method for a
# Population dataclass. we can get this functionality back by creating an
# equivalent staticmethod


def get_aligned_rates(popn, align_ev='stimOn', trange=np.array([[-2, 3]]),
                      binsize=0.05, sm_params={'kind': 'boxcar',
                                               'binsize': 0.05,
                                               'width': 0.4, 'sigma': 0.25},
                      condlabels=['modality', 'coherence', 'heading'],
                      return_Dataset=False):

    good_trs = popn.events['goodtrial'].to_numpy(dtype='bool')
    condlist = popn.events[condlabels].loc[good_trs, :]

    rates = []
    tvecs = []
    align_lst = []

    for al, t_r in zip(align_ev, trange):

        if popn.events.loc[good_trs, al].isna().all(axis=0).any():
            raise ValueError(al)

        align = popn.events.loc[good_trs, al].to_numpy(dtype='float64')

        # get spike counts and relevant t_vec for each unit - thanks chatGPT!
        # trial_psth in list comprehesnsion is going to generate a list of
        # tuples, the zip(*iter) syntax allows us to unpack the tuples into
        # separate variables
        spike_counts, t_vec, _ = \
            zip(*[(trial_psth(unit.spiketimes, align,
                              t_r, binsize, sm_params=sm_params))
                  for unit in popn.units])

        rates.append(np.asarray(spike_counts))
        tvecs.append(np.asarray(t_vec[0]))
        # previously tried dict with key as alignment event

        align_lst.append([al]*len(t_vec[0]))

    align_arr = np.concatenate(align_lst)
    unitlabels = np.array([u.clus_group for u in popn.units])
    # unit_ids  = np.array([u.clus_id for u in popn.units])

    if return_Dataset:
        # now construct a dataset
        arr = xr.DataArray(np.concatenate(rates, axis=2),
                           coords=[unitlabels,
                                   condlist.index,
                                   np.concatenate(tvecs)],
                           dims=['unit', 'trial', 'time'])

        cond_coords = {condlabels[c]: ('trial', condlist.iloc[:, c])
                       for c in range(len(condlist.columns))}

        ds = xr.Dataset({'firing_rate': arr},
                        coords={'align_event': ('time', align_arr),
                                **cond_coords})

        ds.attrs['rec_date'] = popn.rec_date
        ds.attrs['area'] = popn.area

        return ds

    else:
        # return separate vars
        return rates, tvecs, condlist, align_lst