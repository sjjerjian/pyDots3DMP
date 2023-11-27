# %% ----------------------------------------------------------------
# imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import PurePath

from behavior.preprocessing import dots3DMP_create_conditions
from neural.dots3DMP_build_dataset import build_rate_population
from neural.load_utils import load_dataset, quick_load
from neural.NeuralDataClasses import plot_rate, plot_timeseries
from neural.rate_utils import pref_hdg_dir

# %% ===== load data =====

data =  quick_load()
# data = data.iloc[:10] # TEMP, to speed up testing


# %% ----------------------------------------------------------------
# define heading preferences 

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

# %% ----------------------------------------------------------------
# plot just psths

par = 'Task'

cond_labels = ['modality', 'coherenceInd', 'delta', 'heading', 'choice', 'PDW', 'oneTargConf']
# cond_labels = ['modality', 'coherenceInd', 'delta', 'heading', 'correct']

conds_task = {
    'mods': [1, 2, 3],
    'cohs': [1, 2],
    'deltas': [0],
    'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12],
    'choice': [1, 2],
    'PDW': [0, 1],
    'oneTargConf': [0],
}
tr_table = dots3DMP_create_conditions(conds_task, cond_labels)

# t_params = {'align_ev': ['stimOn', 'stimOff'],
#             'trange': np.array([[-2, 0.5], [-0.2, 2]]), 
#             'other_ev': [['fpOn', 'fixation', 'targsOn', 'stimOff'], ['stimOn', 'postTargHold']],
#             'binsize': 0.01,
#             }

t_params = {'align_ev': ['stimOff'],
            'trange': np.array([[-2.5, 2]]), 
            'binsize': 0.05,
            }

sm_params = {'type': 'boxcar',
            'binsize': t_params['binsize'],
            'width': 0.25,
            # 'sigma': 0.05, 
            }




# %% ----------------------------------------------------------------
# plot one unit PSTH

# task_pp = build_rate_population(popns=data[par], tr_table=tr_table,
#                                 t_params=t_params, smooth_params=sm_params,
#                                 event_time_groups=['modality'],
#                                 stacked=False, return_averaged=True)

# sess = 0
# unit = 6

# sess_pop = task_pp[sess].concat_alignments(insert_blank=True)
# sess_pop.reindex_to_event('stimOn')

# unit_fr = sess_pop.firing_rates[unit, :, :]

# plot_timeseries(unit_fr, sess_pop.timestamps, sess_pop.conds,
#                 row='modality', col='coherenceInd', hue='heading',
#                 xlabel='time from stimOn [s]', ylabel='spikes/sec')


    
# %% ----------------------------------------------------------------
# plot population PSTH

task_pp = build_rate_population(popns=data[par], tr_table=tr_table,
                                t_params=t_params, smooth_params=sm_params,
                                event_time_groups=['modality'],
                                stacked=True, return_averaged=True)


# %% ----------------------------------------------------------------
for area in ['MSTd', 'PIVC']:
    area_inds = task_pp.area==area
    area_pp = task_pp.filter_units(inds=area_inds)
    area_pp.concat_alignments(insert_blank=True)
    area_pp.flip_rates(unit_inds=pref_dir_ves[area_inds]==0, col='choice')
    # task_pp.average_rel_event_times().reindex_to_event('stimOn')
    area_pp.demean(standardize=True, return_split='keep')

    sel_conds = (area_pp.conds['heading']==0).values
    conds = area_pp.conds.loc[sel_conds, :]
    conds['choice_wager'] = conds.apply(lambda row: int(2*(row['choice']-1) + row['PDW']), axis=1)

    fig = plot_timeseries(area_pp.firing_rates[:, sel_conds, :], area_pp.timestamps, conds,
                            row='modality', col='coherenceInd', hue='choice_wager',
                            hue_labels=['Null-Lo', 'Null-Hi', 'Pref-Lo', 'Pref-Hi'])
    # fig.set_title(area)
# to do add style


# rate_fig, rate_ax = plot_rate(task_pp.firing_rates, task_pp.timestamps, task_pp.conds, align_events=t_params['align_ev'], 
#                      row='modality', col='coherenceInd', hue='heading')