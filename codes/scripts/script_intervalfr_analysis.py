import numpy as np

import codes.dots3DMP_FRutils as FRutils
from codes.behavior.bhv_preprocessing import dots3DMP_create_trial_list
from codes.dots3DMP_build_dataset import build_pseudopop

import pickle
from pathlib import PurePath

# %% ===== prep the data =====

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = 'lucio_20220512-20230602_neuralData.pkl'
filepath = PurePath(data_folder, 'lucio_neuro_datasets', filename)

with open(filepath, 'rb') as file:
    this_df = pickle.load(file)

par = ['Tuning', 'Task']
data = this_df[this_df[par].notna().all(axis=1)][par]

# %% ===== define conditions of interest =====

expt = 'Task'

cond_labels = ['modality', 'coherenceInd', 'heading', 'delta']
hdgs = [0]
mods = [1, 2, 3]
cohs = [1, 2]
deltas = [0]

if expt == 'Tuning':
    hdgs = [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60]
    align_ev = [['stimOn', 'stimOff'], 'fpOn']
    trange = np.array([[0.5, -0.5], [0, 0.5]])

elif expt == 'Task':
    hdgs = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]
    align_ev = [['fpOn', 'stimOn'], ['stimOn', 'stimOff']]
    trange = np.array([[0, 0], [0, 0]])

tr_tab, _ = dots3DMP_create_trial_list(hdgs, mods, cohs,
                                       deltas, 1, shuff=False)
tr_tab.columns = cond_labels  # TODO use .rename here to make sure ordering is correct

# %% ===== PSTH parameters =====

# set binsize = 0 for gross average/count in each interval
binsize = 0

sm_params = {'type': 'gaussian', 'binsize': binsize,
             'width': 0.4, 'sigma': 0.05}

# ========================================
# %% ===== Extract conditional firing rates for each unit =====
# TODO add code to calc_firing_rates to recalculate tstart and end using third arg (see MATLAB version)

# firing rate, across units and trials, applied to each session separately
f_rates, units, conds, durs, _ = \
    zip(*data[expt].apply(lambda x: x.get_firing_rates(align_ev, trange,
                                                       binsize, sm_params,
                                                       cond_labels)))

# %% ===== Create PseudoPopulation =====

use_single_trials = False

if not use_single_trials:
    # extract condition averaged FRs

    # TODO should refactor these FRutils to operate over concatenated or split frs, possibly with a
    # standard wrapper
    rates_cat, len_intervals = FRutils.concat_aligned_rates(f_rates)

    cond_frs, cond_groups = [], []
    for f_in, cond in zip(rates_cat, conds):

        # avg firing rate over time across units, each cond, per session
        f_out, _, cg = FRutils.condition_averages(f_in, cond, cond_groups=tr_tab)
        cond_frs.append(f_out)
        cond_groups.append(cg)

    # resplit by len_intervals, for pseudopop creation
    f_rates = list(map(lambda f, x: np.squeeze(np.split(f, x, axis=2)[:-1]), cond_frs, len_intervals))

    conds = cond_groups

    # TODO pass in other metadata - e.g. alignments and fr calc parameters (sm params, binsize)

# now create the pseudopopulation
pseudo_pop = build_pseudopop(
    fr_list=f_rates,
    conds_dfs=conds,
    unitlabels=units,
    areas=np.array(data[expt].apply(lambda x: x.area)),
    subject="lucio",
)


np.array([(p=='MSTd') and not low and u==2 for p, low, u in zip(pseudo_pop.area, lf, pseudo_pop.clus_group)]).sum()
np.array([(p=='PIVC') and not low and u==2 for p, low, u in zip(pseudo_pop.area, lf, pseudo_pop.clus_group)]).sum()
