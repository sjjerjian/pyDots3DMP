#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:49:37 2023

@author: stevenjerjian
"""

import pickle
from itertools import repeat
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codes.NeuralDataClasses import Unit, ksUnit, Population, PseudoPop
import codes.dots3DMP_FRutils as FRutils
import codes.tuning_utils as tuning
from codes.behavior.bhv_simulation import dots3DMP_create_trial_list

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = PurePath(data_folder, 'lucio_neuro_datasets',
                    'lucio_20220512-20230602_neuralData.pkl')

with open(filename, 'rb') as file:
    this_df = pickle.load(file)

par = ['Tuning', 'Task']
data = this_df[this_df[par].notna().all(axis=1)][par]

# %% define conditions of interest

# shared for tuning and task
condlabels = ['modality', 'coherenceInd', 'heading', 'delta']
mods = np.array([1, 2, 3])
cohs = np.array([1, 2])
deltas = np.array([0])

# TUNING
hdgs_tuning = np.array([-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60])
tr_tab_tuning, _ = dots3DMP_create_trial_list(hdgs_tuning, mods, cohs,
                                              deltas, 1, shuff=False)
tr_tab_tuning.columns = condlabels

# TASK
hdgs_task = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
tr_tab_task, _ = dots3DMP_create_trial_list(hdgs_task, mods, cohs,
                                            deltas, 1, shuff=False)
tr_tab_task.columns = condlabels


# set binsize = 0 for gross average/count in each interval
binsize = 0.01

# sm_params = {'type': 'boxcar', 'binsize': binsize, 'width': 0.4}
sm_params = {'type': 'gaussian', 'binsize': binsize,
             'width': 0.4, 'sigma': 0.05}

# %% trial firing rates, tuning and task paradigms

# get all unit firing rates across trials/conds, for tuning and task

# TODO add code to calc_firing_rates to recalculate tstart and end
# using third arg (see MATLAB ver)

align_ev = [['stimOn', 'stimOff']]
trange = np.array([[0.5, -0.5]])

# firing rate over time across units and trials, per session
rates_tuning, units_tuning, conds_tuning, tvecs_tuning, _ = \
    zip(*data['Tuning'].apply(lambda x: x.get_firing_rates(align_ev, trange,
                                                            binsize, sm_params,
                                                            condlabels)))

align_ev = ['stimOn']
trange = np.array([[-0.5, 1.5]])

# align_ev = [['fpOn', 'stimOn'], ['stimOn', 'stimOff']]
# trange = np.array([[0, 0], [0, 0]])

rates_task, units_task, conds_task, tvecs_task, _ = \
    zip(*data['Task'].apply(lambda x: x.get_firing_rates(align_ev, trange,
                                                          binsize, sm_params,
                                                          condlabels)))

rel_event_times = data['Task'].apply(lambda x: x.rel_event_times())

# concatenate all intervals
rates_tuning_cat, _ = FRutils.concat_aligned_rates(rates_tuning)
rates_task_cat, _ = FRutils.concat_aligned_rates(rates_task)


# %% cond avg in task, PSTH plotting

cond_frs, cond_groups = [], []
for f_in, cond in zip(rates_task_cat, conds_task):

    # avg firing rate over time across units, each cond, per session
    f_out, _, cg = FRutils.condition_averages(f_in, cond,
                                              cond_groups=tr_tab_task)
    cond_frs.append(f_out)
    cond_groups.append(cg)

# stack 'em up. all units x conditions x time
cond_frs_stacked = np.vstack(cond_frs)


# %% build pseudopop





# %% cond avg + hdg tuning, in TUNING paradigm

tuning_results = {'cond_avgs': {'cond_frs': [],
                                'unique_conds': []},
                  'stats': {'fstats': [],
                            'pvals': [],
                            'unique_conds': []}
                  }

for f_in, cond in zip(rates_tuning_cat, conds_tuning):

    # avg firing rate over time across units, each cond, per session
    f_out, _, cg = FRutils.condition_averages(f_in, cond,
                                              cond_groups=tr_tab_tuning)
    tuning_results['cond_avgs']['cond_frs'].append(f_out)
    tuning_results['cond_avgs']['unique_conds'].append(cg)

    # tuning significance - across headings within mod/coh and interval
    # TODO check this works for multiple intervals

    f_stats, pvals, cg = tuning.tuning_sig(f_in, cond, tr_tab_tuning,
                                           condlabels[:-1])

    tuning_results['stats']['fstats'].append(f_stats)
    tuning_results['stats']['pvals'].append(pvals)
    tuning_results['stats']['unique_conds'].append(cg)

# %% Curation

# TODO clean this up...

# select neurons based on significant tuning
cc = np.array([0, 1, 2, 3, 4])
tuning_sig = list(map(lambda p, c: np.any(p[:, c, 0] < 0.1, axis=1),
                      tuning_results['stats']['pvals'], repeat(cc)))

# drop units with low fr or no significant tuning
bad_tuning = list(map(lambda x: FRutils.lowfr_units(x, 5), rates_tuning_cat))
bad_task = list(map(lambda x: FRutils.lowfr_units(x, 5), rates_task_cat))

bad_units = [np.logical_or(np.logical_or(a, b), sig)
             for a, b, sig in zip(bad_tuning, bad_task, tuning_sig)]

rates_tuning_cat = [fr[~b, :, :] for fr, b in zip(rates_tuning_cat, bad_units)]
rates_task_cat = [fr[~b, :, :] for fr, b in zip(rates_task_cat, bad_units)]
tuning_results['cond_avgs']['cond_frs'] = \
    [fr[~b, :, :] for fr, b in zip(tuning_results['cond_avgs']['cond_frs'],
                                   bad_units)]


# and drop entire 'Population's if <=1 unit
has_units = list(map(lambda fr: fr.shape[0] > 1, rates_tuning_cat))
rates_tuning_cat = [arr for arr, k in zip(rates_tuning_cat, has_units) if k]
rates_task_cat = [arr for arr, k in zip(rates_task_cat, has_units) if k]

tuning_results['cond_avgs']['cond_frs'] = \
    [arr for arr, k in zip(tuning_results['cond_avgs']['cond_frs'],
                           has_units) if k]

conds_tuning = [arr for arr, k in zip(conds_tuning, has_units) if k]
conds_task = [arr for arr, k in zip(conds_task, has_units) if k]

# %% tuning + r_sig

# TODO, fix corr_popn to work separately on each interval/bin, and return corrs
# correspondingly (i.e. independently for each column)

# define preferred heading and direction using condition-averaged FRs
cond_columns = ['modality', 'coherenceInd', 'heading']
dictkeys = ['corrs', 'pvals', 'conds']

sig_corr_results = dict(zip(dictkeys, ([] for _ in dictkeys)))

res = tuning_results['cond_avgs']
for f_in, uconds in zip(res['cond_frs'], res['unique_conds']):

    # split back up, to calculate tuning separately for each interval
    f_in = np.split(f_in, f_in.shape[2], axis=2)

    corrs_t, pvals_t = [], []
    for f_t in f_in:
        pair_corrs, pair_pvals, cg = \
            FRutils.corr_popn(f_t, uconds, tr_tab_tuning, cond_columns,
                              rtype='signal')
        corrs_t.append(pair_corrs)
        pvals_t.append(pair_pvals)

    sig_corr_results['corrs'].append(corrs_t)
    sig_corr_results['pvals'].append(pvals_t)
    sig_corr_results['conds'].append(cg)


# %% spike count/noise correlation (rsc)

# within recording areas

noise_corr_results = dict(zip(dictkeys, ([] for _ in dictkeys)))

for f_in, cond in zip(rates_task_cat, conds_task):

    # split back up, to calculate noise corr separately for each interval
    f_in = np.split(f_in, f_in.shape[2], axis=2)

    corrs_t, pvals_t = [], []
    for f_t in f_in:
        pair_corrs, pair_pvals, cg = \
            FRutils.corr_popn(f_t, cond, tr_tab_task, cond_columns,
                              rtype='noise')
        corrs_t.append(pair_corrs)
        pvals_t.append(pair_pvals)

    noise_corr_results['corrs'].append(corrs_t)
    noise_corr_results['pvals'].append(pvals_t)
    noise_corr_results['conds'].append(cg)


# %% final, plotting

# stacking time
sig_corrs = np.hstack([np.vstack(s[0]) for s in sig_corr_results['corrs']])
noise_corrs = np.hstack([np.vstack(n[0]) for n in noise_corr_results['corrs']])

colors = ['black', 'magenta', 'red', 'cyan', 'blue']
fig, ax = plt.subplots(1, 1)
for c in range(sig_corrs.shape[0]):
    x = sig_corrs[c, :]
    y = noise_corrs[c, :]
    plt.scatter(x, y, color=colors[c], s=10)

# %% noise correlations split by wager

condlabels = ['modality', 'coherenceInd', 'heading', 'delta',
              'correct', 'PDW']
rates_task, ulabs_task, uids_task, tvecs, align_lst, conds_task = \
    zip(*data['Task'].apply(FRutils.get_aligned_rates,
                            args=(align, trange, binsize, sm_params,
                                  condlabels)))

rates_task_PDW, _ = FRutils.concat_aligned_rates(rates_task)

tr_tab_task_lo = tr_tab_task.copy()
tr_tab_task_hi = tr_tab_task.copy()

tr_tab_task_lo['PDW'] = 0
tr_tab_task_hi['PDW'] = 1

tr_tab_task_PDW = pd.concat([tr_tab_task_lo, tr_tab_task_hi], ignore_index=True)
#tr_tab_task_PDW = tr_tab_task_PDW.loc[np.abs(tr_tab_task_PDW['heading'])< 3, :]

tr_tab_task_PDW['correct'] = 1

# drop units with low fr or no significant tuning
bad_tuning = list(map(lambda x: FRutils.bad_rate_units(x, 5), rates_task_PDW))
rates_task_PDW = [fr[~b, :, :] for fr, b in zip(rates_task_PDW, bad_units)]

# and drop entire 'Population's if <=1 unit
has_units = list(map(lambda fr: fr.shape[0] > 1, rates_task_PDW))
rates_task_PDW = [arr for arr, k in zip(rates_task_PDW, has_units) if k]
conds_task = [arr for arr, k in zip(conds_task, has_units) if k]

cond_columns = ['modality', 'PDW', 'correct', 'heading']

noise_corr_PDW = dict(zip(dictkeys, ([] for _ in dictkeys)))

for f_in, cond in zip(rates_task_PDW, conds_task):

    # split back up, to calculate noise corr separately for each interval
    f_in = np.split(f_in, f_in.shape[2], axis=2)

    corrs_t, pvals_t = [], []
    for f_t in f_in:
        pair_corrs, pair_pvals, cg = \
            FRutils.corr_popn(f_t, cond, tr_tab_task_PDW, cond_columns,
                              rtype='noise')
        corrs_t.append(pair_corrs)
        pvals_t.append(pair_pvals)

    noise_corr_PDW['corrs'].append(corrs_t)
    noise_corr_PDW['pvals'].append(pvals_t)
    noise_corr_PDW['conds'].append(cg)

noise_corrs = np.hstack([np.vstack(n[0]) for n in noise_corr_PDW['corrs']])

colors = ['black', 'red', 'blue']  # collapsed across cohs
fig, ax = plt.subplots(1, 1)
plt.axis('square')
for c in range(3):
    x = noise_corrs[c, :]
    y = noise_corrs[c+3, :]
    plt.scatter(x, y, color=colors[c], s=10)
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.7, 0.7)
ax.set_xlabel('Low bet noise corrs')
ax.set_ylabel('High bet noise corrs')


# %% misc scraps

# Drop units not recorded in all conditions
# all_conds = list(map(lambda x: len(x[0]) == 5, noise_corr_results['corrs']))
# for k in noise_corr_results.keys():
#     noise_corr_results[k] = [f for f, keep in zip(noise_corr_results[k], all_conds) if keep]
#     sig_corr_results[k] = [f for f, keep in zip(sig_corr_results[k], all_conds) if keep]

# pref_hdg, pref_dir, pref_hdg_diffs = tuning.tuning_basic(y, hdgs_tuning[~nidx], axis=1)



# plot averages

cond_frs_stacked = np.vstack(tuning_results['cond_avgs']['cond_frs'])

u, a = 0, 0
fr = rates_tuning_cat[0][u, :, a]
df = conds_tuning[0]
df['firing_rate'] = fr

import seaborn as sns
ax = sns.lineplot(data=df,
             x='heading', y='firing_rate',
             estimator='mean', errorbar='se', err_style='bars',
             hue=df[['modality', 'coherenceInd']].apply(tuple, axis=1),
             )
ax.set(xlabel='heading', ylabel = 'firing rate (spikes/sec)')




