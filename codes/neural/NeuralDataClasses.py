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


    # TODO wrap this to allow plotting multiple alignments
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