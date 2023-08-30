#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:37:40 2023

@author: stevenjerjian
"""

import numpy as np
import pandas as pd
import xarray as xr

from datetime import date

from dataclasses import dataclass, field
from rate_utils import *

# %% generic unit class


@dataclass
class Unit:

    spiketimes: np.ndarray = field(repr=False)
    amps: np.ndarray = field(repr=False)

    clus_id: int = field(default=0)
    clus_group: int = 0
    clus_label: str = ''

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

    def mean_wf(self):
        ...

    # def summary(self, binsize=0.01, max=0.2, plot=False):
    #  isi, corr, ifr, wf_width 2x2 subplots


# %% Unit subclass for kilosort data

@dataclass
class ksUnit(Unit):
    """
    a unit extracted from Kilosort, should have a template with amplitude
    also cluster information, and lab/rig specific info
    """

    # TODO
    # wfs: np.ndarray = field(repr=False, default_factory=lambda:
    #    np.zeros(shape=int, dtype=np.float64))
    # template: np.ndarray = field(repr=False, default_factory=lambda:
    #    np.zeros(shape=int, dtype=np.float64))

    unique_id: int = field(init=False)
    temp_amp: float = np.nan

    channel: int = 0
    depth: int = field(default=0, metadata={'unit': 'mm'})
    rec_set: int = 1

    # TODO something doesn't work here - unique_id isn't being recognized as an attribute
    # def __post_init__(self):
    #     self.unique_id = \
    #         int(f"{self.rec_date}{self.rec_set:02d}{self.clus_id:03d}")

    # TODO add contam pct


    def raster_plot(self, align, condlist, col, hue, **kwargs):
        fig, ax = plot_raster(self.spiketimes, align, condlist, col, hue, **kwargs)
        return fig, ax


# %% Population class (simul)


@dataclass
class Population:

    rec_date: date = date.today().strftime("%Y%m%d")
    create_date: date = date.today().strftime("%Y%m%d")
    subject: str = ''
    session: str = ''
    rec_set: int = 1
    probe_num: int = 1
    device: str = ''

    pen_num: int = 1
    grid_xy: tuple[int] = field(default=(np.nan, np.nan))
    grid_type: str = ''
    area: str = ''

    mdi_depth: int = field(default=0, metadata={'unit': 'mm'})
    chs: list[int] = field(default_factory=list, repr=False)
    sr: float = field(default=30000.0, metadata={'unit': 'Hz'})

    units: list[Unit] = field(default_factory=list, repr=False)
    events: dict = field(default_factory=dict, repr=False)
    rel_event_times: dict = field(default_factory=dict, repr=False, init=False)

    def __len__(self):
        return len(self.units)

    def __eq__(self, other):
        if isinstance(other, Population):
            return self.units == other.units
        return False

    def popn_rel_event_times(self, align=['stimOn'], others=['stimOff']):
        self.rel_event_times = rel_event_times(self.events, align, others)
        return self.rel_event_times

    def get_firing_rates(self, *args, **kwargs):
        return calc_firing_rates(self.units, self.events, *args, **kwargs)


@dataclass
class PseudoPop:
    """processed firing rates and events data from one or more recordings"""

    # numpy array units x 'trials'/conditions x time

    # if using trial-averaged data, then 2nd dim should be conditions
    # if using individual trials,
    # and we want to keep additional array labelling trial/unique conditions

    # timestamps

    subject: None
    unit_session: np.ndarray = field(repr=False)
    area: np.ndarray = field(repr=False)
    clus_group: np.ndarray = field(repr=False)

    create_date: date = date.today().strftime("%Y%m%d")

    conds: dict = field(default_factory=dict, repr=False)

    # align_event:  list[str] = field(default_factory=list)
    firing_rates: list[np.ndarray] = field(default_factory=list, repr=False)
    timestamps: list[np.ndarray] = field(default_factory=list, repr=False)

    # TODO add in events, and alignment times!!

    def get_unique_areas(self):
        return np.unique(self.area)

    


# %% util functions


def rel_event_times(events, align=["stimOn"], others=["stimOff"]):

    good_trs = events['goodtrial'].to_numpy(dtype='bool')

    reltimes = {aev: [] for aev in align}
    for aev, oev in zip(align, others):

        if events.loc[good_trs, aev].isna().all(axis=0).any():
            raise ValueError(aev)

        align_ev = events.loc[good_trs, aev].to_numpy(dtype='float64')
        other_ev = events.loc[good_trs, oev].to_numpy(dtype='float64')
        reltimes[aev] = other_ev - align_ev[:, np.newaxis]

    return reltimes


def trial_psth(spiketimes, align, trange = np.array([np.float64, np.float64]),
               binsize=0.05, sm_params=dict,
               all_trials=False, normalize: bool = True):
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
        win /= np.sum(win)  # re-normalize here

    smoothed_fr = convolve1d(raw_fr, win, axis=0, mode='nearest')
    return smoothed_fr


def calc_firing_rates(units, events, align_ev='stimOn', trange=np.array([[-2, 3]]),
                      binsize=0.05, sm_params={},
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
            zip(*[(trial_psth(unit.spiketimes, align, t_r,
                                      binsize, sm_params))
                  for unit in units])

        rates.append(np.asarray(spike_counts))
        tvecs.append(np.asarray(t_vec[0]))

        # this list may be useful if constructing a large pandas dataframe and then using align_event as a hue
        if isinstance(al, str):
            al = [al]
        align_lst.append([al[0]]*len(t_vec[0]))

    align_arr = np.concatenate(align_lst)

    unitlabels = np.array([u.clus_group for u in units])
    # unit_ids  = np.array([u.clus_id for u in popn.units])

    # return an xarray Dataset
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
        # return separate vars
        return rates, unitlabels, condlist, tvecs, align_lst
    

# %% PLOTTING UTILITY FUNCTIONS

# TODO add separate function just for plotting psths, with multiple alignments possible

def plot_raster(spiketimes: np.ndarray, align: np.ndarray, condlist: pd.DataFrame, col: str, hue: str,
                titles=None, suptitle: str = '', align_label='',
                other_evs=None, other_ev_labels=None,  # TODO, not yet implemented
                trange: np.ndarray = np.array([-2, 3]),
                cmap=None, hue_norm=(-12, 12), binsize: int = 0.05, sm_params=None):

    if sm_params is None:
        sm_params = dict()

    df = condlist[col].copy()
    df[hue] = condlist.loc[:, hue]

    # stimOn update to motionOn, time of actual true motion
    if align_label == 'stimOn':
        align += 0.3

    fr, x, df['spks'] = trial_psth(spiketimes, align, trange,
                                           binsize=binsize,
                                           sm_params=sm_params)

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
        ctitle = ', '.join([f'{t}={v:.0f}' for t, v in
                           zip(ctitle.index, ctitle.values)])

        # this time, create groupings based on hue, within cond_df
        ic, nC, cond_groups = condition_index(cond_df[[hue]])

        # need to call argsort twice!
        # https://stackoverflow.com/questions/31910407/
        order = np.argsort(np.argsort(ic)).tolist()

        # TODO further sort within cond by user-specified input e.g. RT

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

        # for seaborn use
        # # set colors for unique headings, with same mapping as raster
        # colors = cmap(hue_norm(np.unique(c_df[hue])).data.astype('float'))
        # colors = np.split(colors, colors.shape[0], axis=0)
        # sns.lineplot(x=c_df['time'], y=c_df['fr'],
        #              hue=hue_group, palette=colors,
        #              ax=ax, legend=False, errorbar=None)

        ax.set_xlim(trange[0], trange[1])
        if c == 0:
            ax.set_ylabel('spikes/sec')
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=hue_norm)
    # sm.set_array([])
    # cbar_ax = fig.add_axes([0.1, 0.1, 0.05, 0.8])
    # cbar = plt.colorbar(sm, ticks=list(uhue))
    cbar = plt.colorbar(sm)
    cbar.set_label(hue)

    # plt.show()

    return fig, axs