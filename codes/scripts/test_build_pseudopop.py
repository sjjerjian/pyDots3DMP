import numpy as np
import pickle as pkl
from pathlib import PurePath

import neural.rate_utils as FRutils
from behavior import dots3DMP_create_trial_list
from neural import build_pseudopop

# %% load the data

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = PurePath(data_folder, 'lucio_neuro_datasets', 'lucio_20220512-20230602_neuralData.pkl')

with open(filename, 'rb') as file:
    this_df = pkl.load(file)

par = ['Tuning', 'Task']
data = this_df[this_df[par].notna().all(axis=1)][par]

# %% define conditions of interest

expt = 'Task'

# shared for tuning and task
condlabels = ['modality', 'coherenceInd', 'heading', 'delta']
mods = np.array([1, 2, 3])
cohs = np.array([1, 2])
deltas = np.array([0])

if expt == 'Tuning':
    hdgs = np.array([-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60])
    align_ev = [['stimOn', 'stimOff'], 'fpOn']
    trange = np.array([[0.5, -0.5], [0, 0.5]])

elif expt == 'Task':
    hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
    align_ev = [['stimOn', 'stimOff'], 'saccOnset']
    trange = np.array([[-1.5, +1.5], [-0.5, 1.5]])

tr_tab, _ = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, 1, shuff=False)
tr_tab.columns = condlabels

# %% get all unit firing rates across trials/conds

# set binsize = 0 for gross average/count in each interval
binsize = 0.05

# sm_params = {'type': 'boxcar', 'binsize': binsize, 'width': 0.4}
sm_params = {
    'type': 'gaussian',
    'binsize': binsize,
    'width': 0.4,
    'sigma': 0.05
}

# TODO add code to calc_firing_rates to recalculate tstart and end using third arg (see MATLAB ver)

# firing rate over time across units and trials, per session
rates, units, conds, tvecs, _ = \
    zip(*data[expt].apply(lambda x: x.get_firing_rates(align_ev, trange, binsize, sm_params, condlabels)))


# pseudo-population with individual trial firing rates

use_single_trials = False

if use_single_trials:
    pseudopop_singletrials = build_pseudopop(rates, units, conds, tvecs)
else:
    # to average firing rates for each condition

    # TODO refactor condition_averages and other FR utils so we don't need to do this??
    # concatenate all the intervals
    rates_cat, len_intervals = FRutils.concat_aligned_rates(rates, tvecs)

    cond_frs, cond_groups = [], []
    for f_in, cond in zip(rates_cat, conds):

        # avg firing rate over time across units, each cond, per session
        f_out, _, cg = FRutils.condition_averages(f_in, cond, cond_groups=tr_tab)
        cond_frs.append(f_out)
        cond_groups.append(cg)

    # resplit by len_intervals, for pseudopop creation
    cond_frs_split = list(map(lambda f, x: np.split(f, x, axis=2)[:-1], cond_frs, len_intervals))

    # pseudo-population with averaged condition firing rates
    # TODO pass in other metadata - e.g. alignments and fr calc parameters (sm params, binsize)
    # also need the relative event timings?

    pseudo_pop = build_pseudopop(
        fr_list=cond_frs_split,
        conds_dfs=cond_groups,
        unitlabels=units,
        areas=np.array(data[expt].apply(lambda x: x.area)),
        subject="lucio",
        tvecs=tvecs,
    )

import matplotlib.pyplot as plt
for al in range(len(pseudo_pop.firing_rates)):

    plt.plot(pseudo_pop.timestamps[al], pseudo_pop.firing_rates[al][0, :, :].T)