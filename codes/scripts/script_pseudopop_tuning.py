# %% imports

import numpy as np

from behavior.preprocessing import dots3DMP_create_trial_list
from neural.rate_utils import build_pseudopop
import pickle
from pathlib import PurePath

# %% ===== load data =====

data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
filename = 'lucio_20220512-20230602_neuralData.pkl'
filename = PurePath(data_folder, 'lucio_neuro_datasets', filename)

with open(filename, 'rb') as file:
    this_df = pickle.load(file)

pars = ['Tuning', 'Task']
data = this_df.loc[this_df[pars].notna().all(axis=1) & this_df['is_good']]

# %% ===== define conditions of interest =====

# these are the same for tuning and task
cond_labels = ['modality', 'coherenceInd', 'heading', 'delta']
mods = [1, 2, 3]
cohs = [1, 2]
deltas = [0]

# ...and headings are different
hdgs_tuning = [-60, -45, -25, -22.5, -12, 0, 12, 22.5, 25, 45, 60]
hdgs_task = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]

# TODO for some sessions headings were different, right now this just ignores those trials

tr_tab_tuning, _ = dots3DMP_create_trial_list(hdgs_tuning, mods, cohs, deltas, 1, shuff=False)
tr_tab_tuning.columns = cond_labels  # should use .rename here to make sure ordering is correct?

tr_tab_task, _ = dots3DMP_create_trial_list(hdgs_task, mods, cohs, deltas, 1, shuff=False)
tr_tab_task.columns = cond_labels  

# %% ===== Create tuning pseudopopulation =====

t_params = {'align_ev': [['stimOn', 'stimOff'], 'fpOn'], 
            'trange': np.array([[0.5, -0.5], [0, 0.5]]), 
            'binsize': 0}

tuning_pseudopop = build_pseudopop(popn_dfs=data['Tuning'], tr_tab=tr_tab_tuning, 
                                   t_params=t_params, return_averaged=True
)

# %% ===== Create task pseudopopulation (time-resolved) =====

t_params = {'align_ev': ['stimOn', 'saccOnset'],
            'trange': np.array([[-1.5, 1.3], [-0.5, 1.5]]), 
            'other_ev': [['fpOn', 'fixation', 'targsOn', 'saccOnset'], ['WagerHold']],
            'binsize': 0.05}

sm_params = {'type': 'gaussian',
             'binsize': t_params['binsize'],
             'width': 0.4, 'sigma': 0.05}

task_pseudopop = build_pseudopop(popn_dfs=data['Task'], tr_tab=tr_tab_task,
    t_params=t_params, smooth_params=sm_params,
    return_averaged=True
    )

# %% ===== Create PseudoPopulation =====

use_single_trials = False

if not use_single_trials:
    # extract condition averaged FRs

    rates_cat, len_intervals = rates.concat_aligned_rates(f_rates, tvecs)

    cond_frs, cond_groups = [], []
    for f_in, cond in zip(rates_cat, conds):

        # avg firing rate over time across units, each cond, per session
        f_out, _, cg = rates.condition_averages(f_in, cond, cond_groups=tr_tab)
        cond_frs.append(f_out)
        cond_groups.append(cg)

    # resplit by len_intervals, for pseudopop creation
    f_rates = list(map(lambda f, x: np.split(f, x, axis=2)[:-1], cond_frs, len_intervals))

    conds = cond_groups

    # TODO pass in other metadata - e.g. alignments and fr calc parameters (sm params, binsize)


# now create the pseudopopulation
pseudo_pop = build_pseudopop(
    fr_list=f_rates,
    conds_dfs=conds,
    unitlabels=units,
    areas=np.array(data[expt].apply(lambda x: x.area)),
    subject="lucio",
    tvecs=tvecs,
)

# import matplotlib.pyplot as plt
# for al in range(len(pseudo_pop.firing_rates)):
#
#     plt.plot(pseudo_pop.timestamps[al], pseudo_pop.firing_rates[al][0, :, :].T)
#


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
