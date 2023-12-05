#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:37:40 2023

@author: stevenjerjian
"""
# %% ----------------------------------------------------------------
# imports

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl

from datetime import date

from dataclasses import dataclass, field
from neural.rate_utils import *
from codetiming import Timer
from collections import defaultdict

from typing import Union, Sequence
from copy import copy, deepcopy

import seaborn as sns
from cycler import cycler
import warnings

# %% ----------------------------------------------------------------
# base unit class

@dataclass
class Unit:

    spiketimes: np.ndarray = field(repr=False)
    rec_date: date = date.today().strftime("%Y%m%d")

    def __len__(self):
        return len(self.spiketimes)

    def isi(self):
        return np.diff(self.spiketimes)

    def isi_cv(self):
        y = self.isi()
        return np.std(y) / np.mean(y)

    def isi_hist(self, binsize=0.02, max_isi=2):
        x = np.arange(0, max_isi+1e-6, binsize)
        y, _ = np.histogram(self.isi(), x, density=True)
        return y, x[:-1]

    def ifr(self):
        return 1 / self.isi()

    def acf(self, binsize=0.02, maxlag=2):
        ...

    def ccf(self, other, binsize=0.02, maxlag=2):
        ...

    def raster_plot(self, align, condlist, col, hue, **kwargs):
        fig, ax = plot_raster(self.spiketimes, align, condlist, col, hue, **kwargs)
        return fig, ax


# %% ----------------------------------------------------------------
# Unit subclass for Kilosort results

@dataclass
class ksUnit(Unit):

    # TODO add templates, amps, waveforms etc

    rec_set: int = 1

    channel: int = 0
    depth: int = field(default=0, metadata={'unit': 'mm'})

    clus_id: int = 0
    clus_group: int = 0
    clus_label: str = ''

    unique_id: int = field(init=False)

    def __post_init__(self):
        self.unique_id = int(f"{self.rec_date}{self.rec_set:02d}{self.clus_id:03d}")


# %% ----------------------------------------------------------------
# Spiking Population class (simultaneously recorded, storing individual spiketimes)

@dataclass
class Population:

    # recording metadata
    rec_date: date = date.today().strftime("%Y%m%d")
    create_date: date = date.today().strftime("%Y%m%d")

    subject: str = ''
    session: str = ''
    rec_set: int = 1
    probe_num: int = 1
    device: str = ''

    # additional metadata, regarding penetration
    pen_num: int = 1
    grid_xy: tuple[int] = field(default=(np.nan, np.nan))
    grid_type: str = ''
    area: str = ''

    mdi_depth: int = field(default=0, metadata={'unit': 'mm'})
    chs: list[int] = field(default_factory=list, repr=False)
    sr: float = field(default=30000.0, metadata={'unit': 'Hz'})

    units: list[Unit] = field(default_factory=list, repr=False)
    events: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.units)

    def __eq__(self, other):
        if isinstance(other, Population):
            return self.units == other.units
        return False

    def popn_rel_event_times(self, align=['stimOn'], others=['stimOff'], cond_groups=None):
        return rel_event_times(self.events, align, others, cond_groups)

    def get_firing_rates(self, *args, **kwargs):
        return calc_firing_rates(self.units, self.events, *args, **kwargs)


#%% ----------------------------------------------------------------
# Population class with binned firing rates (can be a single recording, or multiple (stacked) )

@dataclass
class RatePop:
    """processed firing rates and events data from one or more recordings"""

    subject: str

    # single array, one element per unit
    area: np.ndarray[str] = field(repr=False)
    clus_group: np.ndarray = field(repr=False, metadata={1: 'MU', 2: 'SU'})
    unit_session: np.ndarray = field(default=None, repr=False)

    # one list entry per alignment event
    # each element within firing rates will be units x 'trials'/conditions x time array
    firing_rates: list[np.ndarray] = field(default_factory=list, repr=False)
    timestamps: list[np.ndarray] = field(default_factory=list, repr=False, metadata={'unit':'seconds'})

    sep_alignments: bool = field(default=True, repr=False)
    time_reindexed: bool = field(default=False, repr=False)
    rates_averaged: bool = field(default=False, repr=False)
    simul_recorded: bool = field(default=True, repr=False)

    psth_params: dict = field(default_factory=dict, repr=False)

    # one entry per session
    conds: tuple = field(default_factory=tuple, repr=False)
    rel_events: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    create_date: date = date.today().strftime("%Y%m%d")

    def __len__(self):
        return len(self.clus_group)

    def __repr__(self):
        return f"Rate Pop (Subj: {self.subject}, Areas: {self.get_unique_areas()}, n={len(self)}"

    def get_unique_areas(self):
        return np.unique(self.area).tolist()


    def split(self):
        # TODO allow splitting single RatePop into simulatneously recorded
        if unit_session is not None:
            pass

    def filter_units(self, inds: np.ndarray):
        """
        Filter units in RatePop, using inds
        use cases: extracting single units, extracting tuned units, units from one area

        Args:
            inds (np.ndarray): array of indices of desired units to keep

        Returns:
            filtered_data (RatePop): A sub-selected version of original RatePop, with just the units in inds
        """

        filtered_data = deepcopy(self)

        # one entry per unit
        filtered_data.unit_session = self.unit_session[inds]
        filtered_data.area = self.area[inds]
        filtered_data.clus_group = self.clus_group[inds]
        filtered_data.firing_rates = list(map(lambda x: x[inds, ...], self.firing_rates))

        # one entry per session
        sessions = np.unique(self.unit_session).tolist()

        if isinstance(self.conds, list):
            filtered_data.conds = [self.conds[s] for s in sessions]

        if self.rel_events and isinstance(self.rel_events, list):
             filtered_data.rel_events =  [self.rel_events[s] for s in sessions]

        return filtered_data


    def num_trials_per_unit(self, by_conds=False, cond_groups=None) -> tuple[np.ndarray, pd.DataFrame]:
        # return the number of trials for each unique condition in the population

        self.concat_alignments()

        unit_in_trial = ~np.all(np.isnan(self.firing_rates), axis=2)
        num_units = self.firing_rates.shape[0]

        if by_conds:
            ic, nC, cond_groups = condition_index(self.conds, cond_groups)

            unit_trial_count = np.zeros((num_units, nC))

            for u in range(num_units):
                unit_trial_count[u, :] = np.bincount(ic[(unit_in_trial[u, :]) & (ic>=0)])
        else:
            unit_trial_count = unit_in_trial.sum(axis=1)

        self.split_alignments()

        return unit_trial_count, cond_groups


    def concat_alignments(self, insert_blank: bool = False, warning: bool = False):

        if self.sep_alignments:
            if self.psth_params['binsize'] == 0:
                tvecs = None
                self.concat_ints = len(self.firing_rates)
            else:
                tvecs = self.timestamps

            concat_result = concat_aligned_rates_single(self.firing_rates, tvecs=tvecs, insert_blank=insert_blank)

            self.firing_rates = concat_result[0]
            if self.psth_params['binsize'] > 0:
                self.timestamps = concat_result[1]
                self.concat_ints = concat_result[2]
            self.sep_alignments = False
        else:
            if warning:
                print("Firing rates already a single array, skipping...\n")

        return self


    def split_alignments(self, warning: bool = False):
        if not self.sep_alignments:
            self.firing_rates = np.split(self.firing_rates, self.concat_ints, axis=2)[:-1]

            if self.psth_params['binsize'] > 0:
                self.timestamps = np.split(self.timestamps, self.concat_ints)[:-1]

            self.sep_alignments = True
        else:
            if warning:
                print("Firing rates already a list of alignments, skipping...\n")

        return self

    def demean_across_time(self, t_int=None, t_range=None, across_conds=True, standardize=True, return_split=True):
        """
        t_int: interval to use as baseline - if None, use all (i.e. assume concatenated, or do the concatenation)
        """

        flag_sep = self.sep_alignments

        if return_split == 'keep':
            return_split = flag_sep

        if t_int is None:
            self.concat_alignments()
        else:
            self.split_alignments()

        axis = 2
        if across_conds:
            axis = (2, 1) # average across time, then conditions

        if t_int is None:
            if t_range is None:
                ind0, ind1 = 0, len(self.timestamps)
            else:
                ind0 = np.argmin(np.abs(self.timestamps - t_range[0]))
                ind1 = np.argmin(np.abs(self.timestamps - t_range[1]))


            # subtract average baseline
            bsln_fr = np.nanmean(self.firing_rates[:, :, ind0:ind1], axis=axis)

            if standardize:
                std_divide = np.nanstd(self.firing_rates[:, :, ind0:ind1], axis=2)

                if across_conds:
                    # now take MEAN stdev across conditions
                    std_divide = np.nanmean(std_divide, axis=1)

        else:
            if t_range is None:
                ind0, ind1 = 0, len(self.timestamps[t_int])
            else:
                ind0 = np.argmin(np.abs(self.timestamps[t_int] - t_range[0]))
                ind1 = np.argmin(np.abs(self.timestamps[t_int] - t_range[1]))

            # subtract average baseline
            bsln_fr = np.nanmean(self.firing_rates[t_int][:, :, ind0:ind1], axis=axis)

            if standardize:
                std_divide = np.nanstd(self.firing_rates[t_int][:, :, ind0:ind1], 2)

                if across_conds:
                    # now take MEAN stdev across conditions
                    std_divide = np.nanmean(std_divide, axis=1)

            self.concat_alignments()

        self.firing_rates -= np.expand_dims(bsln_fr, axis=axis)
        if standardize:
            self.firing_rates /= np.expand_dims(std_divide, axis=axis)

        if return_split is True:
            self.split_alignments()

        return self


    def normalize_across_time(self, t_int=None, t_range=None, across_conds: bool = True, softmax_const: int = 0):

        # only separate alignments at end if they were separated to begin with
        flag_sep = self.sep_alignments

        self.demean(t_int, t_range, across_conds=across_conds, standardize=False, return_split=False)

        axis = 2
        if across_conds:
            axis = (1, 2)

        # divide by absolute max across time (and conditions, if across_conds)
        max_fr = np.nanmax(np.abs(self.firing_rates), axis=axis)
        self.firing_rates /= np.expand_dims((max_fr + softmax_const), axis=axis)

        if flag_sep:
            self.split_alignments()

        return self


    def average_rel_event_times(self, by='index'):

        if isinstance(self.rel_events, list):
            print('stop here')

            final_dict = dict()
            for d in self.rel_events:
                for key, df in d.items():
                    if key not in final_dict:
                        final_dict[key] = df
                    else:
                        final_dict[key] = pd.concat([final_dict[key], df])

            # use group-by over index or desired categorical columns
            for key in final_dict.keys():
                if by=='index':
                    grp_by = final_dict[key].index
                else:
                    grp_by = final_dict[key][by]
                final_dict[key] = final_dict[key].groupby(by=grp_by).mean()

            self.rel_events = final_dict

        else:
            print('relative event times are already averaged across sessions, or this is just one session')

        return self


    def reindex_to_event(self, event: str = 'stimOn') -> None:

        if not self.time_reindexed:

            if self.rel_events and event in self.rel_events:

                # self.timestamps_reindexed = copy(self.timestamps)
                time_shift = self.rel_events[event].mean(axis=0).to_dict()

                align_events = [al[0] if isinstance(al, list) else al for al in self.psth_params['align_ev']]

                for t, ev in enumerate(align_events):
                    if ev in time_shift:
                        self.timestamps[t] += time_shift[ev]
                        # self.timestamps_reindexed[t] = self.timestamps[t] + time_shift[ev]
                        print(f"{ev} alignment time base now relative to {event}")
            else:
                print('Unable to re-index - either rel_events does not exist, or an invalid event was provided\n')

            self.time_reindexed = True
        else:
            print('Already re-indexed')


    def recode_conditions(self, columns: Sequence[str], old_values: Sequence[float],
                          new_values: Union[Sequence[float], float]):
        """Recode conditions in order to pool them together for analysis

        Args:
            columns (Sequence): sequence of strings referring to column in conds
            old_values (Sequence): _description_
            new_values (Union[list, int, float]): _description_
        """
        self.conds = recode_conditions(columns, old_values, new_values)


    def flip_rates(self, unit_inds: np.ndarray[Union[bool, int]], col: Union[str, np.ndarray]) -> None:
        """Flip firing rates for specified units for condition in col (e.g. recoding firing rates for right/left as pref/null)

        Args:
            unit_inds (_type_): array of ints or bools referencing which units to flip
            col (Union[str, np.ndarray]): column values for flipping (either a string referring to column in self.conds
            or np.array of hand-coded values. Should have only two unique values)
        """

        if isinstance(col, str):
            col = self.conds[col]

        val_flip = np.unique(col)
        assert len(val_flip) == 2, "Condition for flipping on should have only 2 unique values"

        flag_sep = self.sep_alignments
        self.concat_alignments()

        # flip 'em (just the units selected in unit_inds)

        these_units = self.firing_rates[unit_inds, :, :]
        val0, val1 = col == val_flip[0], col == val_flip[1]

        # this works to simply interchange them, although definitely need some unit tests here to make sure it works properly!
        these_units[:, val0, :], these_units[:, val1, :] = \
            these_units[:, val1, :], these_units[:, val0, :]

        self.firing_rates[unit_inds, :, :] = these_units

        if flag_sep:
            self.split_alignments()



# %% ----------------------------------------------------------------
# externally defined util functions

def recode_conditions(conds: pd.DataFrame, columns: Sequence[str], old_values: Sequence[float],
                      new_values: Union[Sequence[float], float]):
    """Recode conditions in order to pool them together for analysis

    Args:
        conds (pd.DataFrame): dataframe of conditions
        columns (Sequence): sequence of strings referring to columns in conds
        old_values (Sequence): current values of the condition
        new_values (Union[Sequence[float], float]): values to overwrite old_values with

    Returns:
        _type_: _description_
    """

    assert len(columns) == len(old_values)
    if isinstance(new_values, Sequence):
        assert len(new_values) == len(old_values)

    for col in columns:
        for i, orig_val in enumerate(old_values):
            if isinstance(new_values, Sequence):
                self.conds.loc[self.conds[col] == orig_val, col] = new_values[i]
            else:
                self.conds.loc[self.conds[col] == orig_val, col] = new_values

    return conds


def rel_event_times(events: pd.DataFrame, align: Sequence, others: Sequence,
                    cond_groups: Optional[pd.DataFrame]) -> dict:

    good_trs = events['goodtrial'].to_numpy(dtype='bool')

    align_first = [al[0] if isinstance(al, list) else al for al in align]
    reltimes = {k: [] for k in align_first}
    for aev, oev in zip(align_first, others):

        if events.loc[good_trs, aev].isna().all(axis=0).any():
            raise ValueError(aev)

        align_ev = events.loc[good_trs, aev].to_numpy(dtype='float64', na_value=np.nan)
        other_ev = events.loc[good_trs, oev].to_numpy(dtype='float64', na_value=np.nan)
        reltimes[aev] = other_ev - align_ev[:, np.newaxis]

        if cond_groups is not None:
            condlist = events.loc[good_trs, cond_groups.columns]
            ic, nC, cond_groups = condition_index(condlist, cond_groups)

            temp = np.full((nC, reltimes[aev].shape[1]), np.nan)
            for c in range(nC):
                if np.sum(ic == c):
                    temp[c, :] = np.nanmedian(reltimes[aev][ic == c], axis=0)

            reltimes[aev] = temp

        reltimes[aev] = pd.DataFrame(reltimes[aev], columns=oev)

    return reltimes



#@Timer(name='trial_psth_timer', initial_text='trial_psth ')
def trial_psth(spiketimes: np.ndarray, align: np.ndarray,
               trange = np.array([np.float64, np.float64]),
               binsize: float = 0.05,
               sm_params: Optional[dict] = None,
               all_trials: bool = False, normalize: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Args:
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
            binsize for spike count histogram. The default is 0.05 (seconds) i.e. 50ms.
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
        return None

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
        print(f'Dropping {np.sum(nantrs)} trials with missing alignment event')
    align = align[~nantrs, :]
    nTr = align.shape[0]  # recalculate after bad trs removed

    tr_starts = align[:, 0] + trange[0]
    tr_ends = align[:, 1] + trange[1]
    durs = tr_ends - tr_starts

    # compute 'corrected' tStart and tEnd based on align_ev input
    # TODO add explanation for this
    if which_ev == 1:
        tstarts_rel = tr_starts - align[:, 1]
        tstart_rel = np.min(tstarts_rel)
        tend_rel = trange[1]
        tends_rel = tend_rel.repeat(tstarts_rel.shape[0])

    elif which_ev == 0:
        tends_rel = tr_ends - align[:, 0]
        tend_rel = np.max(tends_rel)
        tstart_rel = trange[0]
        tstarts_rel = tstart_rel.repeat(tends_rel.shape[0])

    # reset the absolute tr_starts and tr_ends from here
    tr_starts = align[:, 0] + tstart_rel
    tr_ends = align[:, 1] + tend_rel

    # if smoothing firing rates, extend the range a bit to use real spike counts for smoothing at the edges
    # TODO this is causing a mismatch in lengths later when trying to cut down, need to FIX this
    # if sm_params:
    #     tstart_orig, tend_orig = tstart_rel, tend_rel
    #     tstart_rel -= sm_params['width']/2
    #     tend_rel += sm_params['width']/2

    if binsize > 0:
        if trange[0] < 0 and trange[1] > 0:
            # set forwards and backwards bins separately first, to ensure that time 0 is one of the bin edges
            x0 = np.arange(0, tstart_rel-binsize, -binsize)
            x1 = np.arange(0, tend_rel+binsize+1e-3, binsize)
            x = np.hstack((x0[::-1, ], x1[1:, ]))
        else:
            x = np.arange(tstart_rel, tend_rel+binsize, binsize)

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

                # save the individual times (relative to alignment event) for raster plots
                spktimes_aligned.append(inds_t)

                # set nans outside the range of align/trange for each trial
                if which_ev == 0:
                    end_pos = np.argmin(abs(x - tends_rel[itr]))
                    fr_out[itr, end_pos:] = np.nan
                elif which_ev == 1:
                    start_pos = np.argmin(abs(x - tstarts_rel[itr]))
                    fr_out[itr, :start_pos] = np.nan

        for itr in range(itr_end+1, nTr):
            spktimes_aligned.append(np.empty(1))

        if binsize > 0:
            x = x[:-1] + np.diff(x)/2  # shift 'x' values to bin centers

            if sm_params:
                # TODO allow user to provide custom kernel
                if 'binsize' not in sm_params:
                    sm_params['binsize'] = binsize

                fr_out, sm_nbins = smooth_fr(fr_out, params=sm_params)

                # alternative?
                st = sm_nbins // 2
                en = len(x) - 1 - (st+1)

                # st = np.argmin(np.abs(x - tstart_orig))
                # en = np.argmin(np.abs(x - tend_orig))

                # x = x[st:en]
                # fr_out = fr_out[:, st:en]

            if normalize:
                fr_out /= binsize

        elif binsize == 0:
            if normalize:
                fr_out /= x

    return fr_out, x, spktimes_aligned


def smooth_fr(raw_fr, params: Optional[dict] = None, kernel: Optional[np.ndarray]=None) -> tuple[np.ndarray, int]:

    # TODO set defaults for params even if only some are provided
    if params is None:
        params = {'type': 'boxcar', 'binsize': 0.02, 'width': 0.2, 'normalize': True}

    if kernel is None:
        if params['type'] == 'boxcar':

            N = int(np.ceil(params['width'] / params['binsize']))  # width, in bins
            win = np.ones(N)

            if params['normalize']:
                win /= N

        elif params['type'] == 'gaussian':

            N = int(np.ceil(params['width'] / params['binsize']))  # width, in bins
            alpha = (N - 1) / (2 * (params['sigma'] / params['binsize']))
            win = gaussian(N, std=alpha)

            if params['normalize']:
                win /= np.sum(win)

        elif params['type'] == 'causal':

            raise NotImplementedError('Not implemented yet')

            # width = params['width']
            # if isinstance(width, float):
            #     width = [0.001, width]

            # rise_time, decay_time = width
            # win = np.arange(0, rise_time + decay_time, params['binsize'])
            # win = (1 - np.exp(win / rise_time)) * np.exp(win / decay_time)
            # win /= np.sum(win)  # re-normalize here

    else:
        win = kernel

    # smooth along time, which is axis 1!!!
    smoothed_fr = convolve1d(raw_fr, win, axis=1, mode='nearest')

    return smoothed_fr, N


#@Timer(name='calc_firing_rates_timer')
def calc_firing_rates(units, events, align_ev='stimOn', trange=np.array([[-2, 3]]),
                      binsize: Optional[float] = 0.05, stepsize: Optional[float] = None,
                      sm_params: dict = None,
                      condlabels=('modality', 'coherence', 'heading'),
                      return_ds=False):

    good_trs = events['goodtrial'].to_numpy(dtype='bool')
    condlist = events[condlabels].loc[good_trs, :]

    rates = []
    tvecs = []
    align_lst = []

    for al, t_r in zip(align_ev, trange):

        if events.loc[good_trs, al].isna().all(axis=0).any():
            raise ValueError(al)

        align = events.loc[good_trs, al].to_numpy(dtype='float64')

        # get spike counts and relevant t_vec for each unit
        # trial_psth in list comp is going to generate a list of tuples
        # the zip(*iterable) syntax allows us to unpack the tuples into separate variables
        spike_counts, t_vec, _ = \
            zip(*[(trial_psth(unit.spiketimes, align, t_r, binsize, sm_params)) for unit in units])

        rates.append(np.asarray(spike_counts))
        tvecs.append(np.asarray(t_vec[0]))

        if isinstance(al, str):
            al = [al]
        align_lst.append([al[0]]*len(t_vec[0]))

    align_arr = np.concatenate(align_lst)

    unitlabels = np.array([u.clus_group for u in units])

    # return an xarray Dataset, or separate variables
    if return_ds:
        arr = xr.DataArray(np.concatenate(rates, axis=2),
                           coords=[unitlabels,
                                   condlist.index,
                                   np.concatenate(tvecs)],
                           dims=['unit', 'trial', 'time'])

        cond_coords: dict = {condlabels[c]: ('trial', condlist.iloc[:, c])
                       for c in range(len(condlist.columns))}

        ds = xr.Dataset({'firing_rate': arr},
                        coords={'align_event': ('time', align_arr),
                                **cond_coords})
        return ds

    else:
        return rates, unitlabels, condlist, tvecs, align_lst


# %% PLOTTING UTILITY FUNCTIONS

def plot_raster(spiketimes: np.ndarray, align: np.ndarray, condlist: pd.DataFrame, col: str, hue: str,
                titles=None, suptitle: str = '', align_label: str = '',
                trange: np.ndarray = np.array([-2, 3]),
                cmap=None, hue_norm=(-12, 12), binsize: int = 0.05, sm_params: dict = None):

    # TODO plot other events relative to alignment
    # TODO allow further sort within cond by user-specified input e.g. RT

    if sm_params is None:
        sm_params = dict()

    df = condlist[col].copy()
    df[hue] = condlist.loc[:, hue]

    # stimOn update to motionOn, time of actual true motion
    if align_label == 'stimOn':
        align += 0.3

    fr, x, df['spks'] = trial_psth(spiketimes, align, trange,
                                   binsize=binsize, sm_params=sm_params)

    # if col is a list (i.e. 1 or more conditions), create a new grouping
    # column with unique condition groups. otherwise, just use that column
    if isinstance(col, list):
        ic, nC, cond_groups = condition_index(df[col])
        df['grp'] = ic
    else:
        nC = len(np.unique(df[col]))
        df['grp'] = df[col]

    fig, axs = plt.subplots(nrows=2,  ncols=nC,  figsize=(20, 6))
    fig.suptitle(suptitle)
    fig.supxlabel(f'Time relative to {align_label} [s]')

    # set hue colormap
    uhue = np.unique(df[hue])
    if cmap is None:
        if hue == 'heading':
            cmap = 'RdBu'
        elif hue == 'choice_wager':
            cmap = 'Paired'
        cmap = mpl.colormaps[cmap]

    if isinstance(hue_norm, tuple):
        hue_norm = mpl.colors.Normalize(vmin=hue_norm[0],
                                        vmax=hue_norm[1])
        # hue_norm = mpl.colors.BoundaryNorm(uhue, cmap.N, extend='both')
    # this doesn't work with cmap selection below yet...

    for c, cond_df in df.groupby('grp'):

        ctitle = cond_groups.iloc[c, :]
        ctitle = ', '.join([f'{t}={v:.0f}' for t, v in zip(ctitle.index, ctitle.values)])

        # this time, create groupings based on hue, within cond_df
        ic, nC, cond_groups = condition_index(cond_df[[hue]])

        # need to call argsort twice! https://stackoverflow.com/questions/31910407/
        order = np.argsort(np.argsort(ic)).tolist()

        # get color for each trial, based on heading, convert to list
        colors = cmap(hue_norm(cond_df[hue]).data.astype('float'))
        colors = np.split(colors, colors.shape[0], axis=0)

        # ==== raster plot ====
        # no need to re-order all the lists, just use order for lineoffsets
        ax = axs[0, c]
        ax.eventplot(cond_df['spks'], lineoffsets=order, color=colors)
        ax.set_ylim(-0.5, len(cond_df['spks'])+0.5)
        ax.invert_yaxis()
        ax.set_xlim(trange[0], trange[1])

        if titles is None:
            ax.set_title(ctitle)
        else:
            ax.set_title(titles[c])

        if c == 0:
            ax.set_ylabel('trial')
        else:
            ax.set_yticklabels([])

        # ==== PSTH plot ====
        ax = axs[1, c]

        # get the time-res fr for this condition too
        cond_fr = fr[df['grp'] == c, :]

        cond_colors = cmap(hue_norm(
            cond_groups.loc[:, hue]).data.astype('float'))

        cond_mean_fr = np.full([nC, cond_fr.shape[1]], np.nan)
        cond_sem_fr = np.full([nC, cond_fr.shape[1]], np.nan)

        for hc in range(nC):
            cond_mean_fr[hc, :] = cond_fr[ic == hc, :].mean(axis=0)
            cond_sem_fr[hc, :] = \
                cond_fr[ic == hc, :].std(axis=0) / np.sum(ic == hc)

            ax.plot(x, cond_mean_fr[hc, :], color=cond_colors[hc, :])

        ax.set_xlim(trange[0], trange[1])
        if c == 0:
            ax.set_ylabel('spikes/sec')
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=hue_norm)
    cbar = plt.colorbar(sm)
    cbar.set_label(hue)

    return fig, axs


# %% ----------------------------------------------------------------
# general time series plotting functions


def plot_timeseries(X: np.ndarray, timestamps: Optional[np.ndarray] = None,
                    conds: Optional[pd.DataFrame] = None,
                    xlabel: Optional[str] = '', ylabel: Optional[str] = '',
                    hue_labels: Optional[Sequence] = None,
                    **fig_kws):
    """Plot a timeseries, grouped by conditions (according to fig_kws)
    Specify row, col and hue

    # TODO add style option

    Args:
        X (np.ndarray): array containing time series. conditions/trials x timesteps, or units x conditions/trials x timesteps
        If ndim==3, will average over units (axis=0) first. NOTE that this doesn't do any averaging over trials -
        if X contains condition-averaged data already, those traces will be plotted. If X contains individual trials, individual trials averaged across units will be shown.
        timestamps (Optional[np.ndarray], optional): time indices - will determine xticks. Defaults to None.
        conds (Optional[pd.DataFrame], optional): dataframe containing conditions. Defaults to None.
        xlabel (Optional[str], optional): label for x-axes. Defaults to ''.
        ylabel (Optional[str], optional): label for y-axes. Defaults to ''.

    Returns:
        _type_: figure handle
    """


    if timestamps is None:
        timestamps = np.arange(X.shape[-1])

    if X.ndim == 3:
        X = np.squeeze(np.nanmean(X, axis=0)) # average over units, within each unique trial/condition

    if conds is not None:

        # set hue colormap
        if 'hue' in fig_kws:

            if fig_kws['hue'] == 'heading':
                hue_norm = (-12, 12)
                if isinstance(hue_norm, tuple):
                    hue_norm = mpl.colors.Normalize(vmin=hue_norm[0], vmax=hue_norm[1])
                cmap = 'RdBu'
            elif fig_kws['hue'] == 'choice_wager':
                cmap = 'Paired'

            # FIXME not sure what I was trying to do here...
            cmap = mpl.colormaps[cmap]
            # colors = np.asarray(cmap.colors)
            # colors = colors[:len(np.unique(conds[fig_kws['hue']])), :]

        cond_cols = []
        if 'row' in fig_kws:
            cond_cols.append(fig_kws['row'])
        if 'col' in fig_kws:
            cond_cols.append(fig_kws['col'])

        fig = sns.FacetGrid(data=conds, **fig_kws)
        for ax_key, ax in fig.axes_dict.items():

            cond_trials = (conds[cond_cols]==ax_key).all(axis=1).values
            ax_data = X[cond_trials, :]

            if 'hue' in fig_kws:
                hue_trials = conds.loc[cond_trials, fig_kws['hue']].values
                if fig_kws['hue']=='heading':
                    colors = cmap(hue_norm(hue_trials).data.astype('float'))
                else:
                    pass  # TODO deal with non-heading

                lbl = None
                for hue in np.unique(hue_trials):

                    x_cond_hue = ax_data[hue_trials == hue, :].T
                    x_color = colors[hue_trials == hue, :3]

                    if hue_labels:
                        lbl = hue_labels[hue]
                    ax.plot(timestamps, x_cond_hue, c=x_color, label=lbl)

            else:
                raise NotImplementedError("Not implemented without hue parameter")
                # TODO: allow hue to be ignored

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.legend()

    else:

        fig = plt.figure()


        plt.plot(timestamps, X.T)

    return fig



# WIP
def plot_rate(fr_list: list, timestamps: list, cond_df: pd.DataFrame, align_events, plot_single_trials: bool = False, **fig_kws):

    cond_cols = []
    if 'row' in fig_kws:
        cond_cols.append(fig_kws['row'])
        urow = np.unique(cond_df[fig_kws['row']])
        nrows = len(urow)
    else:
        nrows = 1

    if 'col' in fig_kws:
        cond_cols.append(fig_kws['col'])
        ucol = np.unique(cond_df[fig_kws['col']])
        ncols = len(ucol)
    else:
        ncols = 1

    if len(cond_cols) == 0:
        raise ValueError('must specify row and/or col')

    ncols_final = ncols * len(timestamps)
    width_ratios = ncols * list(map(len, timestamps))

    # set hue colormap
    if 'hue' in fig_kws:

        if fig_kws['hue'] == 'heading':
            hue_norm = (-12, 12)
            if isinstance(hue_norm, tuple):
                hue_norm = mpl.colors.Normalize(vmin=hue_norm[0], vmax=hue_norm[1])
            cmap = 'RdBu'
        elif fig_kws['hue'] == 'choice_wager':
            cmap = 'Paired'

        cmap = mpl.colormaps[cmap]
        colors = cmap

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols_final, gridspec_kw=dict(width_ratios=width_ratios), sharey=True)

    for col in range(ncols_final):
        col_i = col%len(fr_list)
        f_rates = fr_list[col_i]

        if col_i > 0:
            sns.despine(left=True)

        for row in range(nrows):
            sns.despine(right=True, top=True)

            print(row, col, col_i)
            if 'row' in fig_kws and 'col' in fig_kws:
                cond_trials = (cond_df[cond_cols]==(urow[row], ucol[col//len(fr_list)])).all(axis=1).values
            elif 'row' in fig_kws:
                cond_trials = (cond_df[cond_cols]==urow[row]).values
            elif 'col' in fig_kws:
                cond_trials = (cond_df[cond_cols]==ucol[col_i]).values

            hue_trials = cond_df.loc[cond_trials,fig_kws['hue']].values

            if fig_kws['hue']=='heading':
                colors = cmap(hue_norm(hue_trials).data.astype('float'))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                ax_data = np.squeeze(np.nanmean(f_rates[:, cond_trials, :], axis=0)) # average over units

                for hue in np.unique(hue_trials):

                    y_data = ax_data[hue_trials == hue, :].T
                    color = colors[hue_trials == hue, :]

                    if not plot_single_trials:
                        y_data = np.nanmean(y_data, axis=1)
                        color = np.nanmean(color, axis=0)

                    axs[row, col].plot(timestamps[col_i], y_data, color=color)

            axs[row, col].set_xlim([timestamps[col_i][0], timestamps[col_i][-1]])

            xtix = axs[row, col].get_xticks()
            xtix_new = np.where(xtix==0, align_events[col_i], xtix)
            axs[row, col].set_xticks(xtix, xtix_new)

    plt.show()
                      
    return fig, axs