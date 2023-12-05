#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:17:18 2023

@author: stevenjerjian
"""



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
# Aggregated interval activity during task, for defining preferred directions/choices during task

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
