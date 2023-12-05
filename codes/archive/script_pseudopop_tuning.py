# %% ----------------------------------------------------------------
# imports 

import numpy as np
import pandas as pd
import pickle
from pathlib import PurePath

from behavior.preprocessing import dots3DMP_create_trial_list, add_trial_outcomes
from neural.rate_utils import build_pseudopop, concat_aligned_rates
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
cond_labels = ['modality', 'coherenceInd', 'delta', 'heading']
mods = [1, 2, 3]
cohs = [1, 2]
deltas = [0]

# headings are different...
hdgs_tuning = [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60]
hdgs_task = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]

# TODO for some sessions headings were different, right now this just ignores those trials

tr_tab_tuning = dots3DMP_create_trial_list(hdgs_tuning, mods, cohs, deltas, 1, shuff=False)
tr_tab_tuning.columns = cond_labels  # should use .rename here to make sure ordering is correct?

tr_tab_task = dots3DMP_create_trial_list(hdgs_task, mods, cohs, deltas, 1, shuff=False)
tr_tab_task.columns = cond_labels  

# %% ========================
# Create tuning pseudopopulation 

# t_params = {'align_ev': [['stimOn', 'stimOff'], 'fpOn'], 
#             'trange': np.array([[0.5, -0.5], [0, 0.5]]), 
#             'binsize': 0}

# tuning_pseudopop = build_pseudopop(popn_dfs=data['Tuning'], tr_tab=tr_tab_tuning, 
#                                    t_params=t_params, return_averaged=True,
# )

# %% ========================
# Calculating tuning significance (across headings and across baseline/stimulus)

rates_tuning, units_tuning, conds_tuning, _, _ = \
    zip(*data['Tuning'].apply(lambda x: x.get_firing_rates(align_ev=t_params['align_ev'],
                                                           trange=t_params['trange'],
                                                           binsize=t_params['binsize'],
                                                           condlabels=cond_labels)
                                                           ))

rates_tuning_cat, _ = concat_aligned_rates(rates_tuning)

tuning_res_within = {'stats': [], 'pvals': [], 'unique_conds': []}
tuning_res_across = {'stats': [], 'pvals': [], 'unique_conds': []}
for f_in, cond in zip(rates_tuning_cat, conds_tuning):

    stats, pvals, cg = tuning_across(f_rates=f_in, condlist=cond, cond_groups=tr_tab_tuning,
                                     cond_cols=['modality', 'coherenceInd'], tuning_col='heading',
                                     bsln_t=1, abs_diff=True, parametric=True)
    
    tuning_res_across['stats'].append(stats)
    tuning_res_across['pvals'].append(pvals)
    tuning_res_across['unique_conds'].append(cg)
    
    stats, pvals, cg = tuning_within(f_rates=f_in, condlist=cond, cond_groups=tr_tab_tuning,
                                     cond_cols=['modality', 'coherenceInd'], tuning_col='heading',
                                     parametric=True)
    
    tuning_res_within['stats'].append(stats)
    tuning_res_within['pvals'].append(pvals)
    tuning_res_within['unique_conds'].append(cg)


tuning_conds = tuning_res_across['unique_conds'][0]
pvals_across = np.vstack(tuning_res_across['pvals'])
pvals_within = np.vstack(tuning_res_within['pvals'])

# %% ----------------------------------------------------------------

# for each unit, take single trial firing rates, then  for eachcondition do roc analysis across binary split by another var (choice, PDW)


# %% ----------------------------------------------------------------
area = np.array(tuning_pseudopop.area)
print(f"MSTd n={np.sum(area=='MSTd')}, PIVC n={np.sum(area=='PIVC')}")
MSTsu = np.sum((area=='MSTd') & (tuning_pseudopop.clus_group==2))
PIVCsu = np.sum((area=='PIVC') & (tuning_pseudopop.clus_group==2))

print(f"MSTd n={MSTsu}, PIVC n={PIVCsu}")


# np.array([(p=='MSTd') and not low and u==2 for p, low, u in zip(pseudo_pop.area, lf, pseudo_pop.clus_group)]).sum()
# np.array([(p=='PIVC') and not low and u==2 for p, low, u in zip(pseudo_pop.area, lf, pseudo_pop.clus_group)]).sum()



# %% ----------------------------------------------------------------
# # ===== Create task  pseudopopulation (time-resolved) =====

t_params = {'align_ev': ['stimOn', 'saccOnset'],
            'trange': np.array([[-1.5, 1.3], [-0.5, 1.5]]), 
            'other_ev': [['fpOn', 'fixation', 'targsOn', 'saccOnset'], ['postTargHold']],
            'binsize': 0.05}

sm_params = {'type': 'gaussian',
             'binsize': t_params['binsize'],
             'width': 0.4, 'sigma': 0.05}

# task_pseudopop = build_pseudopop(popn_dfs=data['Task'], tr_tab=tr_tab_task,
#     t_params=t_params, smooth_params=sm_params, return_averaged=True)


# %% ========================

# TODO ANALYSES

# tuning of each unit during task
# significance of heading tuning in task, variation from baseline, confidence and choice separation
# 'choice probability', 'wager probability'
# 
# regression onto stimulus/decision variables (need acc/vel from nexonar) also use fixation, fixation point, targets, stimulus conditions, choice and wager outcomes (1/-1)
# construct matrix of regressors
# this is on trial-averages, aligned to stimOn or saccOnset


# import matplotlib.pyplot as plt
# for al in range(len(pseudo_pop.firing_rates)):
#
#     plt.plot(pseudo_pop.timestamps[al], pseudo_pop.firing_rates[al][0, :, :].T)
#




# %% ========================
# save pseudo-pop to file, if desired

print(f"Pseudopopulation built for {filename}")

save_file = PurePath(data_folder, 'lucio_neuro_datasets',
                     "lucio_20220512-20230602_pseudopop.pkl")
with open(save_file, 'wb') as file:
    pickle.dump(pseudo_pop, file)




# %% ===== some checks

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
save_file = PurePath(data_folder, 'lucio_neuro_datasets',
                     "lucio_20220512-20230602_pseudopop.pkl")
with open(save_file, 'rb') as file:
    pseudo_pop = pickle.load(file)
