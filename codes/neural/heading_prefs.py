

# define tuning preferences

import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import PurePath
from typing import Optional

from behavior.preprocessing import dots3DMP_create_conditions
from neural.dots3DMP_build_dataset import build_rate_population
from neural.load_utils import load_dataset

def tuning_heading_preference(data, par):
    
    conds_tuning = {
        'mods': [1, 2],
        'hdgs': [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60],
    }
    tr_tuning = dots3DMP_create_conditions(conds_tuning)

    t_params = {'align_ev': [['stimOn', 'stimOff']],
                'trange': np.array([[0.5, -0.5]]), 
                'binsize': 0}

    tuning_pp = build_rate_population(popns=data[par], tr_table=tr_tuning,
                                    t_params=t_params, stacked=True, return_averaged=True)

    pref_dirs, pref_frs = pref_hdg_dir(tuning_pp.firing_rates[0], tuning_pp.conds, tuning_pp.conds, method='sum')
    
    return pref_dirs, pref_frs


def get_modality_prefs(pref_dirs, pref_frs):

    # define pref_dir in terms of vestibular preference, visual prference, or the stronger of the two
    pref_dir_ves = pref_dirs[0, :]
    pref_dir_vis = pref_dirs[1, :]

    pref_mod = np.argmax(np.abs(pref_frs[:, :2]), axis=1)
    pref_dir_any = pref_dirs[np.indices(pref_mod.shape)[0], pref_mod]
    
    return pref_dir_ves, pref_dir_vis, pref_dir_any



if __name__ == '__main__':
    
    filename_date = 'lucio_20220512-20230602'
    par = 'Tuning'
    
    data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data/lucio_neuro_datasets/'
    data = load_dataset(f'{filename_date}_neuralData.pkl', data_folder, pars=par)
    
    pref_dirs, pref_frs = tuning_heading_preference(data, par)
    pref_dir_ves, pref_dir_vis, pref_dir_any = get_modality_prefs(pref_dirs, pref_frs)
    
    # TO DO save the results to file
    