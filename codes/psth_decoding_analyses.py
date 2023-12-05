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

from behavior.preprocessing import dots3DMP_create_conditions
from neural.dots3DMP_build_dataset import build_rate_population
from neural.load_utils import quick_load
from neural.rate_utils import pref_hdg_dir, demean_conditions, condition_averages
from neural.decoding import decode_classifier, decode_roc

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold

from neural.NeuralDataClasses import plot_timeseries


# %% ----------------------------------------------------------------
# define heading preference using tuning task?

def tuning_heading_preference(data):

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
    pref_dir_ves = pref_dirs[:, 0]
    pref_dir_vis = pref_dirs[:, 1]

    pref_mod = np.argmax(np.abs(pref_frs[:, :2]), axis=1)
    pref_dir_any = pref_dirs[np.indices(pref_mod.shape)[0], pref_mod]

    return pref_dir_ves, pref_dir_vis, pref_dir_any


# %%

data = quick_load()
data = data.iloc[:3]  # TEMP, to speed up testing

# _, enough_trials = check_num_trials_per_condition()
pref_dir_ves, pref_dir_vis, pref_dir_any = tuning_heading_preference(data)

# %% ----------------------------------------------------------------
# Build time-resolved pseudo-population for task

conds_cw = {
    'mods': [1, 2, 3],
    'cohs': [1, 2],
    'deltas': [0],
    'hdgs': [0], # don't use super easy headings
    'choice': [1, 2],
    #'PDW': [0, 1],
    'oneTargConf': [0],
}
cond_labels_cw = ['modality', 'coherenceInd', 'delta', 'heading', 'choice', 'oneTargConf']
tr_choice_wager = dots3DMP_create_conditions(conds_cw, cond_labels_cw)

# fine time-resolution, smoothed bin counts
t_params = {'align_ev': ['stimOn', 'saccOnset'],
            'trange': np.array([[-1.5, 1], [-0.5, 1.5]]),
            'other_ev': [['fpOn', 'fixation', 'targsOn', 'saccOnset'], ['stimOn', 'postTargHold']],
            'binsize': 0.01
            }

sm_params = {'type': 'boxcar',
            'binsize': t_params['binsize'],
            'width': 0.25,
            'normalize': True,
            #'sigma': 0.05,
            }

# coarse time-resolution - averages within intervals
t_params = {'align_ev': ['targsOn', ['stimOn', 'saccOnset'], 'saccOnset', 'postTargHold'],
            'trange': np.array([[-0.4, 0.0], [0, 0], [-0.3, 0.1], [-0.1, 0.3]]),
            'binsize': 0,
            }


# task_pops_trials  = build_rate_population(popns=data['Task'], tr_table=tr_choice_wager,
#                                 t_params=t_params, smooth_params=sm_params,
#                                 event_time_groups=['modality'],
#                                 stacked=False, return_averaged=False)


task_pops_trials  = build_rate_population(popns=data['Task'], tr_table=tr_choice_wager,
                                t_params=t_params, stacked=False, return_averaged=False)

# condition averages, for testing
task_pops_avgs  = build_rate_population(popns=data['Task'], tr_table=tr_choice_wager,
                                t_params=t_params, stacked=True, return_averaged=True)
task_pops_avgs.concat_alignments()

# %% Demean firing rates over these conditions


conds_demean = {k: conds_cw[k] for k in conds_cw.keys() & {'mods', 'cohs', 'deltas', 'hdgs'}}
cond_labels = ['modality', 'coherenceInd', 'delta', 'heading']
tr_tab_demean = dots3DMP_create_conditions(conds_demean, cond_labels)

# %% Decoders for Choice, Wager

# base object for logistics
model = LogisticRegression(penalty='l1', solver='liblinear')
# sgkf = StratifiedGroupKFold(n_splits=5)


choice_logit_scores, wager_logit_scores = [], []
cp_scores, wp_scores = [], []

sess_cond_frs, sess_demeaned_frs, sess_cw_frs = [], [], []

for sess_num, pp in enumerate(task_pops_trials):

    print(sess_num)
    # pp.reindex_to_event('stimOn') # TODO issue warning if binsize is 0
    pp.concat_alignments(insert_blank=True)

    # subtract from each trial the mean of its stimulus condition (across all choice and wager outcomes)
    demeaned_frs, cond_frs, cond_groups = demean_conditions(pp.firing_rates, pp.conds[cond_labels], cond_groups=tr_tab_demean, standardize=False)

    #cw_avg_frs, _, _ = condition_averages(pp.firing_rates, pp.conds[cond_labels_cw], cond_groups=tr_choice_wager)

    sess_cond_frs.append(cond_frs)
    sess_demeaned_frs.append(demeaned_frs)
    #sess_cw_frs.append(cw_avg_frs)

    # choice_logit_score, cg = decode_classifier(f_rates=demeaned_frs, condlist=pp.conds, cond_groups=tr_choice_wager,
    #                                            cond_cols=['modality', 'coherenceInd'], outcome_col='choice',
    #                                            model=model, decode_as_population=True, cv=5)
    # choice_logit_scores.append(choice_logit_score)

    # wager_logit_score, cg = decode_classifier(f_rates=demeaned_frs, condlist=pp.conds, cond_groups=tr_choice_wager,
    #                                            cond_cols=['modality', 'coherenceInd'], outcome_col='PDW',
    #                                            model=model, decode_as_population=True, cv=5)
    # wager_logit_scores.append(wager_logit_score)

    # %% repeat for CPs/confPs

    # using zero heading only with raw firing rates
    # sel_conds = (pp.conds['heading']==0).to_numpy(dtype='bool')
    # conds_zerohdg = pp.conds.loc[sel_conds, :]
    # tr_tab_zerohdg = tr_choice_wager.loc[tr_choice_wager['heading']==0, :]
    # cp_score, cg = decode_roc(f_rates=pp.firing_rates, condlist=conds_zerohdg, cond_groups=tr_tab_zerohdg,
    #                                     cond_cols=['modality'], outcome_col='choice')

    # Choice Probability ===
    cp_score, cg = decode_roc(f_rates=pp.firing_rates, condlist=pp.conds, cond_groups=tr_choice_wager,
                              cond_cols=['modality', 'coherenceInd'], outcome_col='choice')
    cp_scores.append(cp_score)

    # === Wager/Confidence Probability ===
    # wp_score, cg = decode_roc(f_rates=demeaned_frs, condlist=pp.conds, cond_groups=tr_choice_wager,
    #                           cond_cols=['modality', 'coherenceInd'], outcome_col='PDW')
    # wp_scores.append(wp_score)


# %%

# flip CPs to preferred/null, rather than right/left
cp_stacked = np.vstack(cp_scores)
cp_stacked[np.isnan(cp_stacked)] = 0.5

cp_stacked = np.where(cp_stacked > 0.5, cp_stacked, 1-cp_stacked)



# %% drop units with low firing rates



# %%

# plt.hist(cp_stacked.flatten())
# cp_stacked[pref_dir_vis==0, :, :] = 1 - cp_stacked[pref_dir_vis==0, :, :]
# cp_stacked[cp_stacked<0.5] == 1 - cp_stacked[cp_stacked<0.5]
# plt.hist(cp_stacked.flatten())

# plt.plot(pp.timestamps, cp_stacked.mean(axis=0).T)


# %%

# sess_cond_frs = np.vstack(sess_cond_frs)
# plot_timeseries(sess_cond_frs, conds=cond_groups, row='modality', col='coherenceInd', hue='heading')
