# conditional PSTHs

# 1. Number of trials per condition for each unit (for exclusion criteria, task and/or tuning)
# 2. Average firing rates across conditions for each unit (for exclusion criteria, task and/or tuning)
# 3. PSTH (+ raster?) plots for each unit for different condition groupings
# 4. Choice/Wager Probabilities
# 5. Tuning to heading in tuning task
# 6. Tuning to heading
# 7. Modulation across time (timing of peak modulation?)
# 8. Logistic Regression/Support Vector Machine to predict choice/wager/heading dir
# 9. TDR (Regression) analysis to evaluate encoding of different task variables
# 10. PCA with 'RT' axis (see Chand, Remington papers?)
# 11. (all of the above, split by area?)

# %% ----------------------------------------------------------------
# load data (and general imports)

import numpy as np
import pandas as pd

from pathlib import PurePath

from behavior.preprocessing import dots3DMP_create_conditions
from neural.dots3DMP_build_dataset import build_rate_population
from neural.load_utils import load_dataset, quick_load
from neural.rate_utils import pref_hdg_dir
from neural.decoding import decode_outcome

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------

def main():
    data =  quick_load()
    # data = data.iloc[:4] # TEMP, to speed up testing

    # _, enough_trials = check_num_trials_per_condition()

    # pref_dir_ves, pref_dir_vis, pref_dir_any = tuning_heading_preferences()

    task_pp_single_trials, tr_choice_wager = build_pseudopop_cond_psth(data, stacked=False, averaged=False)
    
    model = LogisticRegression(penalty='l1', solver='liblinear')
    # sgkf = StratifiedGroupKFold(n_splits=5)
    
    logit_decode_scores, cp_scores, wp_scores = [], [], []
    for pp in task_pp_single_trials:
        pp.demean(return_split=False)
        
        # Logistic decoder
        # logit_decode_score, cg = decode_outcome(f_rates=pp.firing_rates, condlist=pp.conds, cond_groups=tr_choice_wager, 
        #                                         cond_cols=['modality'], outcome_col='choice', estimator=model,
        #                                         decode_as_population=True)
        # logit_decode_scores.append(logit_decode_score)  
        
        # Choice Probability
        sel_conds = (pp.conds['heading']==0).to_numpy(dtype='bool')
        conds = pp.conds.loc[sel_conds, :]
        
        tr_tab = tr_choice_wager.loc[tr_choice_wager['heading']==0, :]
    
        cp_score, cg = decode_outcome(f_rates=pp.firing_rates[:, sel_conds, :], condlist=conds, cond_groups=tr_tab, 
                                           cond_cols=['modality'], outcome_col='choice', estimator='ROC')
        cp_scores.append(cp_score)  
        
        # Wager/Confidence Probability
        wp_score, cg = decode_outcome(f_rates=pp.firing_rates, condlist=pp.conds, cond_groups=tr_choice_wager, 
                                           cond_cols=['modality'], outcome_col='PDW', estimator='ROC')
        wp_scores.append(wp_score)  
                                
                                
    # recode firing rates to preferred/null, rather than right/left
    # NOTE: no need to do this for choice probabilities, because we can just flip after the fact (1-CP if left preferring)
    # task_pp.flip_rates(pref_dir_any==0, 'choice')
    print('stop here')


# %% ----------------------------------------------------------------
# check num trials per condition per unit 

def check_num_trials_per_condition(min_tr=5):
    
    conds_task = {
        'mods': [1, 2, 3], 'cohs': [1, 2], 'deltas': [0],
        'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12],
    }
    cond_labels = ['modality', 'coherenceInd', 'delta', 'heading']
    tr_task = dots3DMP_create_conditions(conds_task, cond_labels)

    t_params = {'align_ev': [['stimOn', 'stimOff']],
                'trange': np.array([[0.5, -0.5]]), 
                'binsize': 0}

    task_pp = build_rate_population(popns=data['Task'], tr_table=tr_task,
                                    t_params=t_params, stacked=False, return_averaged=False)

    unit_trial_counts = []
    enough_trials = []
    for p in task_pp:
        utc, _ = p.num_trials_per_unit(by_conds=True, cond_groups=tr_task)
        unit_trial_counts.append(utc)
        enough_trials.append((utc >= min_tr).all(axis=1))
    enough_trials = np.hstack(enough_trials)
    
    return task_pp, enough_trials


# %% ----------------------------------------------------------------
# define heading preference using tuning task?

def tuning_heading_preference():
    
    conds_tuning = {
        'mods': [1, 2],
        'hdgs': [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60],
    }
    tr_tuning = dots3DMP_create_conditions(conds_tuning)

    t_params = {'align_ev': [['stimOn', 'stimOff']],
                'trange': np.array([[0.5, -0.5]]), 
                'binsize': 0}

    tuning_pp = build_rate_population(popns=data['Tuning'], tr_table=tr_tuning,
                                    t_params=t_params, stacked=True, return_averaged=True)

    pref_dirs, pref_frs = pref_hdg_dir(tuning_pp.firing_rates[0], tuning_pp.conds, tuning_pp.conds, method='sum')

    # define pref_dir in terms of vestibular preference, visual prference, or the stronger of the two
    pref_dir_ves = pref_dirs[0, :]
    pref_dir_vis = pref_dirs[1, :]

    pref_mod = np.argmax(np.abs(pref_frs[:, :2]), axis=1)
    pref_dir_any = pref_dirs[np.indices(pref_mod.shape)[0], pref_mod]
    
    return pref_dir_ves, pref_dir_vis, pref_dir_any

# %% ----------------------------------------------------------------
# Build time-resolved pseudo-population for task

def build_pseudopop_cond_psth(data, stacked, averaged):
    conds_task = {
        'mods': [1, 2, 3],
        'cohs': [1, 2],
        'deltas': [0],
        'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12],
        'choice': [1, 2],
        'PDW': [0, 1],
        'oneTargConf': [0],
    }
    cond_labels = ['modality', 'coherenceInd', 'delta', 'heading', 'choice', 'PDW', 'oneTargConf']
    tr_choice_wager = dots3DMP_create_conditions(conds_task, cond_labels)

    # t_params = {'align_ev': ['stimOn', 'saccOnset'],
    #             'trange': np.array([[-1.5, 1.3], [-0.5, 1.5]]), 
    #             'other_ev': [['fpOn', 'fixation', 'targsOn', 'saccOnset'], ['stimOn', 'postTargHold']],
    #             'binsize': 0.01
    #             }
    
    t_params = {'align_ev': ['stimOff'],
                'trange': np.array([[-2.5, 2.5]]), 
                'other_ev': [['fpOn', 'fixation', 'targsOn', 'saccOnset']],
                'binsize': 0.01
                }

    sm_params = {'type': 'gaussian',
                'binsize': t_params['binsize'],
                'width': 0.4,
                'sigma': 0.05, 
                }

    task_pp = build_rate_population(popns=data['Task'], tr_table=tr_choice_wager,
                                    t_params=t_params, smooth_params=sm_params,
                                    event_time_groups=['modality'],
                                    stacked=stacked, return_averaged=averaged)

    return task_pp, tr_choice_wager



# %% ----------------------------------------------------------------
# Aggregated interval activity during task, for defining preferred directions/choices during task

# 200ms from fixation, stimulus period, and saccade to wager hold
# t_params = {'align_ev': ['fixation', ['stimOn', 'saccOnset'], ['saccOnset', 'postTargHold']],
#             'trange': np.array([[0, 0.2], [0, 0], [0, 0]]), 
#             'binsize': 0}

# # need non-averaged activity (individual trials)
# task_pp0 = build_rate_population(popns=data['Task'], tr_tab=tr_choice_wager,
#                                  t_params=t_params, stacked=True, return_averaged=False)


# task_pp.reindex_to_event('stimOn')
# task_pp.normalize(t_bsln=(0, [-0.5, 0]), softmax_const=5).concat_alignments(insert_blank=True)

# f_rates = task_pp.firing_rates
# tvecs = task_pp.timestamps
# condlist = task_pp.conds[0]

# # assign a unique 'outcome' based on choice & PDW (1 of 4)
# condlist['outcome'] = condlist.apply(lambda row: row['choice'] + 2*row['PDW'], axis=1)

# condlist = condlist.loc[np.abs(condlist['heading'])<2]
# f_rates = f_rates[:, condlist.index, :]

# row, col, hue = 'modality', 'coherenceInd', 'outcome'
# g = sns.FacetGrid(data=condlist, row=row, col=col, hue=hue)

# for ax_key, ax in g.axes_dict.items():

#     cond_inds = (condlist[row] == ax_key[0]) & (condlist[col] == ax_key[1])
    
#     c_cond = condlist.loc[cond_inds, :]
#     f_cond = f_rates[:, cond_inds, :]
    
#     left_prefs = f_cond[pref_dir_any == -1, :, :]
#     left_prefs[:, c_cond['choice'].values==1, :], left_prefs[:, c_cond['choice'].values==2, :] = \
#         left_prefs[:, c_cond['choice'].values==2, :], left_prefs[:, c_cond['choice'].values==1, :]
#     f_cond[pref_dir_any == -1, :, :] = left_prefs
    
#     f_cond = np.nanmean(f_cond, axis=0)
    
#     for h in np.unique(c_cond[hue]):
#         f_hue = f_cond[c_cond[hue]==h, :]
#         ax.plot(tvecs, np.nanmean(f_hue, axis=0)) # collapse again across other variables
    


if __name__ == '__main__':
    main()