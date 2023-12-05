# %% ========================
# imports 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import PurePath

from behavior.preprocessing import dots3DMP_create_conditions
from neural.dots3DMP_build_dataset import build_rate_population
from neural.load_utils import load_dataset, quick_load

# %% ===== load data =====

data =  quick_load()
data = data.iloc[:2] # TEMP, to speed up testing

# %% ===== set trial table =====

cond_labels = ['modality', 'coherenceInd', 'heading']
conds_tuning = {
        'mods': [1, 2],
        'cohs': [1, 2],
        'hdgs': [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60],
    }
tr_table = dots3DMP_create_conditions(conds_tuning, cond_labels)

cond_labels = ['modality', 'coherenceInd', 'delta', 'heading', 'choice', 'PDW', 'oneTargConf']
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

# %% ----------------------------------------------------------------
# example raster and psth

sess, unit = 0, 0  # session & unit
align_ev = 'stimOn'

events = data['Task'][sess].events
good_trs = ((events['goodtrial'] & ~events['oneTargConf'])).to_numpy(dtype='bool')
condlist = events[cond_labels].loc[good_trs, :]

align = events.loc[good_trs, align_ev].to_numpy(dtype='float64')

titles = ['Ves', 'Vis L', 'Vis H', 'Comb L', 'Comb H']

binsize = 0.05
sm_params = {'type': 'boxcar', 'binsize': binsize, 'width': 0.4}

rh_fig, rh_ax = data['Task'][sess].units[unit].raster_plot(
    align, condlist, ['modality', 'coherenceInd'], 'heading', titles,
    align_ev, trange=np.array([-3, 5]), binsize=binsize, sm_params=sm_params)

# %% ----------------------------------------------------------------
# all rh plots

align_ev = 'stimOn'
thisPar = 'Task'
this_par_data = data[thisPar]

t_params = {'align_ev': 'stimOn',
            'trange': np.array([-3, 5]), 
            'other_ev': ['fpOn', 'fixation', 'targsOn', 'saccOnset', 'postTargHold'],
            'binsize': 0.05,
            }

sm_params = {'type': 'boxcar',
            'binsize': t_params['binsize'],
            'width': 0.4,
            }

plt.ioff()
with PdfPages('allunits_rasterhist.pdf') as pdf_file:

    for sess in this_par_data:
        good_trs = ((sess.events['goodtrial'] & ~sess.events['oneTargConf'])).to_numpy(dtype='bool')
        condlist = sess.events[cond_labels].loc[good_trs, :]

        align = sess.events.loc[good_trs, t_params['align_ev']].to_numpy(dtype='float64')

        for unit in sess.units:
            unit_title = f'{unit.rec_date}, set={unit.rec_set}, id={unit.clus_id}, group={unit.clus_group}'

            if len(unit):
                rh_fig, rh_ax = unit.raster_plot(align, condlist,
                                                 ['modality', 'coherenceInd'], 'heading', 
                                                 unit_title, align_ev, trange=t_params['trange'],
                                                 binsize=t_params['binsize'], sm_params=sm_params)
                pdf_file.savefig(rh_fig)


