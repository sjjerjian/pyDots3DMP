# %% ----------------------------------------------------------------
# imports 

import numpy as np
import pandas as pd

from behavior.preprocessing import dots3DMP_create_trial_list, add_trial_outcomes
from neural.dots3DMP_build_dataset import build_rate_population
from neural.rate_utils import roc_outcome, pref_hdg_dir
from neural.tuning_utils import tuning_across, tuning_within

# %% ----------------------------------------------------------------
# load data 

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = 'lucio_20220512-20230602_neuralData.pkl'
filename = PurePath(data_folder, 'lucio_neuro_datasets', filename)

with open(filename, 'rb') as file:
    data = pd.read_pickle(file)

pars = ['Tuning', 'Task']
data = data.loc[data[pars].notna().all(axis=1) & data['is_good']]
# data = data.iloc[:2] # TEMP, to speed up testing/dev when needed

# %% ----------------------------------------------------------------
# Neural population firing rates from tuning paradigm
# binsize=0 so we just get the average rate in stimulus interval

mods = [1, 2, 3]
deltas = [0]

# for defining heading preference
hdgs_tuning = [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60]
cohs = [1]
tr_tuning = dots3DMP_create_trial_list(hdgs_tuning, mods, cohs, deltas, 1, shuff=False)
tr_tuning.columns = ['modality', 'coherenceInd', 'delta', 'heading']
tr_tuning.drop(labels='coherenceInd', axis=1, inplace=True) # drop coherence # TODO modify create_trial_list to allow not specifying coherence from the beginning?

t_params = {'align_ev': [['stimOn', 'stimOff']],
            'trange': np.array([[0.5, -0.5]]), 
            'binsize': 0}

tuning_pp = build_rate_population(popn_dfs=data['Tuning'], tr_tab=tr_tuning,
                                  t_params=t_params, stacked=True, return_averaged=True)
pref_dirs, pref_frs = pref_hdg_dir(tuning_pp.firing_rates[0], tuning_pp.conds[0], method='sum')
    
# define pref_dir in terms of vestibular preference, visual, or the stronger of the two
pref_dir_ves = pref_dirs[:, 0]
pref_dir_vis = pref_dirs[:, 1]
pref_mod = np.argmax(np.abs(pref_frs[:, :2]), axis=1)
pref_dir_any = pref_dirs[np.indices(pref_mod.shape)[0], pref_mod]



# %%

# now recreate tuning_population but with individual trials
# for defining significant tuning
# TODO, should have a way to go from one directly to the other after creation..
# i.e. to stack/unstack a population, and average a population of single trials
tuning_pp = build_rate_population(popn_dfs=data['Tuning'], tr_tab=tr_tuning,
                                  t_params=t_params, stacked=False, return_averaged=False)

# TODO
# for p in tuning_pp:
#     p.concat_alignments()
#     f_stat, p_val, cg = tuning_within(p.firing_rates, p.conds, tr_tuning)
    
# %% ----------------------------------------------------------------
# Task Paradigm

mods = [1, 2, 3]
cohs = [1, 2]
deltas = [0]
hdgs_task = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]

hdgs_cp = [-1.5, 0, 1.5] 
hdgs_cp = [0]

outcomes = {'choice': [1, 2], 'PDW': [0, 1], 'oneTargConf': [0]}

tr_task = dots3DMP_create_trial_list(hdgs_task, mods, cohs, deltas, 1, shuff=False) # for simple stimulus tuning
tr_task.columns = ['modality', 'coherenceInd', 'delta', 'heading']

tr_choice_wager = dots3DMP_create_trial_list(hdgs_cp, mods, cohs, deltas, 1, shuff=False).pipe(add_trial_outcomes, outcomes)
tr_choice_wager.columns = ['modality', 'coherenceInd', 'delta', 'heading', 'choice', 'PDW', 'oneTargConf']

t_params = {'align_ev': ['fpOn', 'stimOn', 'saccOnset', 'postTargHold'],
            'trange': np.array([[0, 0.2], [0.3, 0.6], [-0.3, 0], [-0.3, 0]]), 
            'binsize': 0}

task_pseudopop_for_tuning = build_rate_population(popn_dfs=data['Task'], tr_tab=tr_task,
                                                    t_params=t_params, stacked=False, return_averaged=False)


all_fstats, all_pvals = [], []
for p in task_pseudopop_for_tuning:
    p.concat_alignments()
    f_stat, p_val, cg = tuning_within(p.firing_rates, p.conds, tr_tuning)
    all_fstats.append(f_stat)
    all_pvals.append(p_val)
    
# could/should split population instances by area first...
MST = [p for i, p in enumerate(all_pvals) if areas[i]=='MSTd']
PIVC = [p for i, p in enumerate(all_pvals) if areas[i]=='PIVC']

MST_stacked = np.vstack(MST)
PIVC_stacked = np.vstack(PIVC)

np.sum(MST_stacked<0.01, axis=0)
np.sum(PIVC_stacked<0.01, axis=0)

# %% ----------------------------------------------------------------
# for choice probability calculations, we need single trial firing rates

t_params = {'align_ev': ['fpOn', 'stimOn', 'saccOnset', 'postTargHold'],
            'trange': np.array([[0, 0.2], [0.3, 0.6], [-0.3, 0], [-0.3, 0]]), 
            'binsize': 0}

task_pseudopop = build_rate_population(popn_dfs=data['Task'], tr_tab=tr_choice_wager,
                                       t_params=t_params, stacked=False, return_averaged=False)

# trials_per_unit = task_pseudopop[0].num_trials_per_unit(by_conds=True, cond_groups=tr_choice_wager)

# %% ----------------------------------------------------------------


# TODO address deprecationWarning with ragged nested sequences
pop_sizes = np.array(list(map(lambda x: x.firing_rates[0].shape[0], task_pseudopop)))

pref_dirs = np.split(pref_dirs, pop_sizes.cumsum(), axis=0)
pref_dir_ves = np.split(pref_dir_ves, pop_sizes.cumsum(), axis=0)
pref_dir_vis = np.split(pref_dir_vis, pop_sizes.cumsum(), axis=0)
pref_dir_any = np.split(pref_dir_any, pop_sizes.cumsum(), axis=0)

cond_cols = ['modality', 'heading']
choice_probs, wager_probs = [], []
choice_conds, wager_conds = [], []

for p, d in zip(task_pseudopop, pref_dir_any):

    p.concat_alignments()
    p.recode_conditions(['heading'], [-1.5, 1.5], 0)
    
    # select tuned units, etc
    # p.filter_units()
    
    c_prob, cg = roc_outcome(p.firing_rates, condlist=p.conds, cond_groups=tr_choice_wager,
                 cond_cols=cond_cols, outcome_col='choice', pos_label=d)
    choice_probs.append(c_prob)
    choice_conds.append(cg)  # cg should be the same for every session/unit
    
    w_prob = roc_outcome(p.firing_rates, condlist=p.conds, cond_groups=tr_choice_wager,
                 cond_cols=cond_cols, outcome_col='PDW')
    wager_probs.append(w_prob)
    wager_conds.append(cg)

cp_allunits = np.vstack(choice_probs)
wp_allunits = np.vstack(wager_probs)
