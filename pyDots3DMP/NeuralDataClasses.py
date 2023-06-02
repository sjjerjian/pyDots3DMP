#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:37:40 2023

@author: stevenjerjian
"""

import numpy as np
import xarray as xr

import pdb

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import date

from dataclasses import dataclass, field
import dots3DMP_FRutils as FRutils

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

    # something about this doesn't work, doesn't recognize as attribute
    # def __post_init__(self):
    #     self.unique_id = \
    #         int(f"{self.rec_date}{self.rec_set:02d}{self.clus_id:03d}")

    # TODO add contam pct
    # TODO allow this to plot multiple alignments?

    def raster_plot(spiketimes, align, condlist, col, hue, **kwargs):
        fig, ax = plot_raster(spiketimes, align, condlist, col, hue, **kwargs)
        return fig, ax


# %% Population class (simul)


@dataclass
class Population:
    """
    data from one recording set
        - metadata
        - list of unit instances, one per recorded unit
        - dictionary/pandas df of task events and times
    """

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

    def __len__(self):
        return len(self.units)

    def __eq__(self, other):
        if isinstance(other, Population):
            return self.units == other.units
        return False

    def popn_rel_event_times(self, align=['stimOn'], others=['stimOff']):
        return rel_event_times(self.events, align, others)

    def calc_firing_rates(self, align_ev='stimOn', trange=np.array([[-2, 3]]),
                          binsize=0.05, sm_params={},
                          condlabels=['modality', 'coherence', 'heading'],
                          return_Dataset=False):

        good_trs = self.events['goodtrial'].to_numpy(dtype='bool')
        condlist = self.events[condlabels].loc[good_trs, :]

        rates = []
        tvecs = []
        align_lst = []

        for al, t_r in zip(align_ev, trange):

            if self.events.loc[good_trs, al].isna().all(axis=0).any():
                raise ValueError(al)

            align = self.events.loc[good_trs, al].to_numpy(dtype='float64')

            # get spike counts and relevant t_vec for each unit
            # trial_psth in list comp is going to generate a list of tuples
            # the zip(*iter) syntax allows us to unpack the tuples into the
            # separate variables
            spike_counts, t_vec, _ = \
                zip(*[(FRutils.trial_psth(unit.spiketimes, align, t_r,
                                          binsize, sm_params))
                      for unit in self.units])

            rates.append(np.asarray(spike_counts))
            tvecs.append(np.asarray(t_vec[0]))

            # considered dict align_lst with key as alignment event, but al[0]
            # might not be unique!

            align_lst.append([al]*len(t_vec[0]))

        align_arr = np.concatenate(align_lst)

        unitlabels = np.array([u.clus_group for u in self.units])
        # unit_ids  = np.array([u.clus_id for u in popn.units])

        if return_Dataset:
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

            ds.attrs['rec_date'] = self.rec_date
            ds.attrs['area'] = self.area

            return ds

        else:
            # return separate vars
            return rates, unitlabels, condlist, tvecs, align_lst


# %% PseudoPop class


@dataclass
class PseudoPop:
    """processed firing rates and events data from one or more recordings"""

    # numpy array units x 'trials' x time
    # note that trials can be replaced by conditions here,
    # or trials x conditions combination
    # (although if we have individual trials across non-simul recorded,
    # the trial label is meaningless)

    # timestamps

    clus_group: np.ndarray = field(repr=False)

    create_date: date = date.today().strftime("%Y%m%d")
    subject: str = ''
    area: str = ''

    hist_sm_params: dict = field(default_factory=dict, repr=False)
    hist_binsize: float = 0

    conds: dict = field(default_factory=dict, repr=False)

    align_event:  list[str] = field(default_factory=list)
    firing_rates: list[np.ndarray] = field(default_factory=list, repr=False)
    timestamps: list[np.ndarray] = field(default_factory=list, repr=False)

# %%


def rel_event_times(events, align=['stimOn'], others=['stimOff']):

    good_trs = events['goodtrial'].to_numpy(dtype='bool')

    reltimes = {aev: [] for aev in align}
    for aev, oev in zip(align, others):

        if events.loc[good_trs, aev].isna().all(axis=0).any():
            raise ValueError(aev)

        align_ev = events.loc[good_trs, aev].to_numpy(dtype='float64')
        other_ev = events.loc[good_trs, oev].to_numpy(dtype='float64')
        reltimes[aev] = other_ev - align_ev[:, np.newaxis]

    return reltimes


def plot_raster(spiketimes, align, condlist, col, hue,
                titles=None, suptitle='',
                align_label='', other_evs=[], other_ev_labels=[],
                trange=np.array([-2, 3]),
                cmap=None, hue_norm=(-12, 12),
                binsize=0.05, sm_params={}):

    # condlist should be pandas df with conditions
    # align_ev should be np array of same length as condlist

    df = condlist[col].copy()
    df[hue] = condlist.loc[:, hue]

    # stimOn update to motionOn, time of actual true motion
    if align_label == 'stimOn':
        align += 0.3

    fr, x, df['spks'] = FRutils.trial_psth(spiketimes, align, trange,
                                           binsize=binsize,
                                           sm_params=sm_params)

    # if col is a list (i.e. 1 or more conditions), create a new grouping
    # column with unique condition groups. otherwise, just use that column
    if isinstance(col, list):
        ic, nC, cond_groups = FRutils.condition_index(df[col])
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
        ic, nC, cond_groups = FRutils.condition_index(cond_df[[hue]])

        # need to call argsort twice!
        # https://stackoverflow.com/questions/31910407/
        order = np.argsort(np.argsort(ic)).tolist()

        # TODO further sort within cond by user input e.g. RT

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
