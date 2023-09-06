# %% ========================
# imports 

import numpy as np
import pandas as pd

from pathlib import PurePath

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# custom imports
from behavior.preprocessing import dots3DMP_create_trial_list
import neural.rate_utils as rates


# %% ===== load data =====

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = 'lucio_20220512-20230602_neuralData.pkl'
filename = PurePath(data_folder, 'lucio_neuro_datasets', filename)

with open(filename, 'rb') as file:
    data = pd.read_pickle(file)

par = ['Tuning', 'Task']
data = data[data[par].notna().all(axis=1)][par]  # drop sessions without par

# %% ===== set trial table =====

condlabels = ['modality', 'coherenceInd', 'heading', 'delta']
mods = np.array([1, 2, 3])
cohs = np.array([1, 2])
deltas = np.array([0])

# hdgs = np.array([-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60])
hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
tr_tab, _ = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas,
                                       1, shuff=False)
tr_tab.columns = condlabels


# %% example raster and psth

sess, unit = 0, 0  # session, unit
align_ev = 'stimOn'

events = data['Task'][sess].events
good_trs = ((events['goodtrial'] & ~events['oneTargConf'])).to_numpy(dtype='bool')
condlist = events[condlabels].loc[good_trs, :]

align = events.loc[good_trs, align_ev].to_numpy(dtype='float64')

titles = ['Ves', 'Vis L', 'Vis H', 'Comb L', 'Comb H']

binsize = 0.05
sm_params = {'type': 'boxcar', 'binsize': binsize, 'width': 0.4}

rh_fig, rh_ax = data['Task'][sess].units[unit].plot_raster(
    align, condlist, ['modality', 'coherenceInd'], 'heading', titles,
    align_ev, trange=np.array([-3, 5]), binsize=binsize, sm_params=sm_params)

# %% all rh plots

align_ev = 'stimOn'
thisPar = 'Task'

this_par_data = data[thisPar]
this_par_data = this_par_data[1:]

binsize = 0.05
sm_params = {'type': 'boxcar', 'binsize': binsize, 'width': 0.4}

plt.ioff()
with PdfPages('allunits_rasterhist.pdf') as pdf_file:

    for sess in this_par_data:
        good_trs = ((sess.events['goodtrial'] & ~sess.events['oneTargConf'])).to_numpy(dtype='bool')
        condlist = sess.events[condlabels].loc[good_trs, :]

        align = sess.events.loc[good_trs, align_ev].to_numpy(dtype='float64')

        for unit in sess.units:
            unit_title = f'{unit.rec_date}, set={unit.rec_set}, id={unit.clus_id}, group={unit.clus_group}'

            if len(unit):
                rh_fig, rh_ax = unit.plot_raster(align, condlist,
                                                 ['modality', 'coherenceInd'],
                                                 'heading', unit_title,
                                                 align_ev,
                                                 trange=np.array([-3, 5]),
                                                 binsize=binsize,
                                                 sm_params=sm_params)
                pdf_file.savefig(rh_fig)


