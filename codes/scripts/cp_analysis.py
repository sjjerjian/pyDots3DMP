# %% ----------------------------------------------------------------
# imports 

import numpy as np
import pandas as pd
import pickle
from pathlib import PurePath

from behavior.preprocessing import dots3DMP_create_trial_list, add_trial_outcomes
from neural.rate_utils import build_pseudopop, concat_aligned_rates, outcome_prob
from neural.tuning_utils import *

# %% ----------------------------------------------------------------
# load data 

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = 'lucio_20220512-20230602_neuralData.pkl'
filename = PurePath(data_folder, 'lucio_neuro_datasets', filename)

with open(filename, 'rb') as file:
    data = pd.read_pickle(file)

pars = ['Tuning', 'Task']
data = data.loc[data[pars].notna().all(axis=1) & data['is_good']]

# %% ----------------------------------------------------------------
# define conditions of interest 

# these are the same for tuning and task
cond_labels = ['modality', 'coherenceInd', 'delta', 'heading', 'choice', 'PDW', 'oneTargConf']
mods = [1, 2, 3]
cohs = [1, 2]
deltas = [0]
# hdgs_task = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]
hdgs_task = [-1.5, 0, 1.5] 

outcomes = {'choice': [1, 2], 'PDW': [0, 1], 'oneTargConf': [0]}
tr_tab_task = dots3DMP_create_trial_list(
    hdgs_task, mods, cohs, deltas, 1, shuff=False
    ).pipe(add_trial_outcomes, outcomes)
tr_tab_task.columns = cond_labels  


# %% ----------------------------------------------------------------
# for choice probability calculations, we need single trial firing rates

t_params = {'align_ev': ['stimOn', 'saccOnset'],
            'trange': np.array([[-1.5, 1.3], [-0.5, 1.5]]), 
            'other_ev': [['fpOn', 'fixation', 'targsOn', 'saccOnset'], ['postTargHold']],
            'binsize': 0}

sm_params = {'type': 'gaussian',
             'binsize': t_params['binsize'],
             'width': 0.4, 'sigma': 0.05}

rates_task, units_task, conds_task, tvecs_task, _ = \
    zip(*data['Task'].apply(lambda x: x.get_firing_rates(align_ev=t_params['align_ev'],
                                                         trange=t_params['trange'],
                                                         binsize=t_params['binsize'],
                                                         condlabels=cond_labels)
                                                         ))


rates_cat, len_intervals = concat_aligned_rates(rates_task)

# %% ----------------------------------------------------------------
choice_probs, wager_probs = [], []
#choice_conds, wager_conds = [], []

for f_in, cond in zip(rates_cat, conds_task):
    
    c_prob, cg = outcome_prob(f_in, condlist=cond, cond_groups=tr_tab_task,
                 cond_cols=['modality','coherenceInd','heading'], outcome_col='choice')
    choice_probs.append(c_prob)
    #choice_conds.append(cg)  # cg is the same for every session/unit
    
    w_prob = outcome_prob(f_in, condlist=cond, cond_groups=tr_tab_task,
                 cond_cols=['modality','coherenceInd','heading'], outcome_col='PDW')
    wager_probs.append(w_prob)
    #wager_conds.append(cg)

cp_allunits = np.vstack(choice_probs)
wp_allunits = np.vstack(wager_probs)

# task_pseudopop = build_pseudopop(popn_dfs=data['Task'], tr_tab=tr_tab_task,
    # t_params=t_params, smooth_params=sm_params, return_averaged=True)

# %%
