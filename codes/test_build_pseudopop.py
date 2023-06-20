


import pickle as pkl
from itertools import repeat
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codes.NeuralDataClasses import Unit, ksUnit, Population, PseudoPop
import codes.dots3DMP_FRutils as FRutils
import codes.tuning_utils as tuning
from codes.dots3DMP_behavior import dots3DMP_create_trial_list
from codes.dots3DMP_build_dataset import build_pseudopop

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = PurePath(data_folder, 'lucio_neuro_datasets',
                    'lucio_20220512-20230602_neuralData.pkl')

with open(filename, 'rb') as file:
    this_df = pkl.load(file)

par = ['Tuning', 'Task']
data = this_df[this_df[par].notna().all(axis=1)][par]

# %% define conditions of interest

# shared for tuning and task
condlabels = ['modality', 'coherenceInd', 'heading', 'delta']
mods = np.array([1, 2, 3])
cohs = np.array([1, 2])
deltas = np.array([0])

# TUNING
hdgs_tuning = np.array([-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60])
tr_tab_tuning, _ = dots3DMP_create_trial_list(hdgs_tuning, mods, cohs,
                                              deltas, 1, shuff=False)
tr_tab_tuning.columns = condlabels


# set binsize = 0 for gross average/count in each interval
binsize = 0.01

# sm_params = {'type': 'boxcar', 'binsize': binsize, 'width': 0.4}
sm_params = {'type': 'gaussian', 'binsize': binsize,
             'width': 0.4, 'sigma': 0.05}

# %% trial firing rates, tuning and task paradigms

# get all unit firing rates across trials/conds, for tuning and task

# TODO add code to calc_firing_rates to recalculate tstart and end
# using third arg (see MATLAB ver)

align_ev = [['stimOn', 'stimOff'], 'fpOn']
trange = np.array([[0.5, -0.5], [0, 0.5]])

# firing rate over time across units and trials, per session
rates_tuning, units_tuning, conds_tuning, tvecs_tuning, _ = \
    zip(*data['Tuning'].apply(lambda x: x.get_firing_rates(align_ev, trange,
                                                            binsize, sm_params,
                                                            condlabels)))

# %%
build_pseudopop(rates_tuning, units_tuning, conds_tuning, tvecs_tuning)

