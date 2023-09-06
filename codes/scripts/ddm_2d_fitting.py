# %% imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from ddm_moi.Accumulator import AccumulatorModelMOI
from ddm_moi import ddm_2d

from behavior.preprocessing import dots3DMP_create_trial_list, data_cleanup, format_onetargconf
from behavior.descriptive import *

from pybads import BADS
#from scipy.optimize import minimize

# %% ===== generate (and save) simulated data =====

sim_params = {'kmult': 0.15, 'bound': np.array([1, 1]), 'alpha': 0.05, 'theta': [0.8, 0.6, 0.7],
                'ndt': [0.1, 0.3, 0.2], 'sigma_ndt': 0.06, 'sigma_dv': 1}
        
mods = np.array([1, 2, 3])
cohs = np.array([0.3, 0.7])
hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
deltas = np.array([-3, 0, 3])

nreps = 200
trial_table, ntrials = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=False)

accum = AccumulatorModelMOI(tvec=np.arange(0, 2, 0.005), grid_vec=np.arange(-3, 0, 0.025))
sim_data, _ = ddm_2d.generate_data(sim_params, data=trial_table, 
                                         accumulator=accum, method='simulate', return_wager=True)

# TODO, option to save save sim_data as csv
with open(f"../data/sim_behavior_202308_{nreps}reps.pkl", "wb") as file:
    pickle.dump((sim_data, sim_params), file, protocol=-1)


# %% ===== load simulated data =====

# nreps = 200
# filename = f"../../data/sim_behavior_202308_{nreps}reps.pkl" # replace with real data
# with open(f"../../data/sim_behavior_202308_{nreps}reps.pkl", "rb") as file:
#     data, sim_params = pickle.load(file)

# %% ===== load real monkey data =====

filename = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230606.csv"

data = data_cleanup(filename)
data = format_onetargconf(data, remove_one_targ=True)

# initialize accumulator
accum = AccumulatorModelMOI(tvec=np.arange(0, 2, 0.05), grid_vec=np.arange(-3, 0, 0.025))

# %% ===== set parameters for optimization =====

# fit zero conflict only
data_delta0 = data.loc[data['delta']==0, :]

# first fitting choice and RT only
init_params = {'kmult': 0.3, 'bound': np.array([0.8, 0.8]), 'ndt': [0.15, 0.25, 0.2], 'sigma_ndt': 0.06}
init_params_array = ddm_2d.get_params_array_from_dict(init_params)

fixed = np.ones_like(init_params_array)
# fixed[0:3] = 0  # fitting kmult and bound only
fixed[:-1] = 0 # fitting all except sigma_ndt

# set bounds and plausible bounds
# lb, ub = init_params_array * 0.1, init_params_array * 10
# plb, pub = init_params_array * 0.3, init_params_array * 5

lb = np.array([0.05, 0.3, 0.3, 0.05, 0.05, 0.05, 0])
ub = np.array([3,  2, 2, 0.5, 0.5, 0.5, 0.15])

plb = np.array([0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.03])
pub = np.array([2,  1.5, 1.5, 0.3, 0.4, 0.3, 0.1])

# %% ===== RUN IT =====

target = lambda params: ddm_2d.objective(params, init_params, fixed,
                                         data=data_delta0, accumulator=accum,
                                         outputs=['choice', 'RT'], llh_scaling=[1, 1])

# fit using Bayesian Adaptive Direct Search (BADS)
fit_options = {'random_seed': 42} # for testing/reproducibility
# eventually initialize multiple times from different starting points
bads = BADS(target, init_params_array, lb, ub, plb, pub, options=fit_options)
bads_result = bads.optimize()  

# using scipy optimize minimize
#args_tuple = (init_params, fixed, param_keys, data, accum, ['choice', 'RT'], [1, 0.1])
#fit_options = {'disp': True, 'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True}
# min_result = minimize(ddm_2d.objective, init_params_array, method='Nelder-Mead',
#                args=args_tuple, options=fit_options)


# TODO make a simple loop to re-run with multiple starting points?

# %% ===== re-run optimization, but now with PDW =====
# rerun (with tight bounds on all already fitted parameters, except alpha and theta)

x = np.array([0.50296439, 0.6763085, 0.67342862, 0.3947464 , 0.5, 0.45350879, 0.06])

init_params2 = ddm_2d.set_params_dict_from_array(x, init_params)
init_params2['sigma_ndt'] = 0.06
init_params2['theta'] = [0.4, 0.3, 0.4]
init_params2['alpha'] = 0.05
init_params_array = ddm_2d.get_params_array_from_dict(init_params2)

# again, only sigma_ndt is fixed
fixed = np.zeros_like(init_params_array)
fixed[6] = 1

# now set bound constraints tightly for the params we already fit
lb = np.array([0.4, 0.6, 0.6, 0.3, 0.4, 0.3, 0, 0, 0, 0, 0])
ub = np.array([0.6, 0.75, 0.75, 0.55, 0.6, 0.55, 0.15, 1, 1, 1, 0.15])

plb = np.array([0.45, 0.65, 0.65, 0.35, 0.45, 0.4, 0.03, 0.2, 0.2, 0.2, 0.02])
pub = np.array([0.55, 0.72, 0.72, 0.45, 0.55, 0.5, 0.1, 0.8, 0.8, 0.8, 0.1])

target_wpdw = lambda params: ddm_2d.objective(params, init_params2, fixed,
                                    data=data_delta0, accumulator=accum,
                                    outputs=['choice', 'PDW', 'RT'], llh_scaling=[1, 1, 0.1])

# fit using Bayesian Adaptive Direct Search (BADS)
fit_options = {'random_seed': 42} # for testing/reproducibility
# eventually initialize multiple times from different starting points
bads_wpdw = BADS(target_wpdw, init_params_array, lb, ub, plb, pub, options=fit_options)
bads_result_wpdw = bads_wpdw.optimize()  

# %% now take the fitted parameters, and generate predictions

# 
# fitted_params = {
#     'kmult': 0.350975, 'bound': np.array([0.970, 0.967]),
#     'alpha': 0.05, 'theta': [0.8, 0.6, 0.7],
#     'ndt': [0.465], 'sigma_ndt': 0.06,
#     }

# fitted_params = {
#     'kmult': 0.412, 'bound': np.array([0.6, 0.6]),
#     'alpha': 0.064, 'theta': [1, 0.845, 0.961],
#     'ndt': [0.423, 0.6, 0.480], 'sigma_ndt': 0.06,
#     }

# fitted_params = ddm_2d.set_params_dict_from_array(bads_result.x, init_params)
# fitted_params['sigma_ndt'] = 0.06

fitted_params = {
    'kmult': 0.462, 'bound': np.array([0.651, 0.652]),
    'ndt': [0.397, 0.5, 0.45], 'sigma_ndt': 0.06,
    }

# generate conditions list for prediction
num_hdgs = 20              # to plot a smooth curve of model prediction over headings
nreps = 1                   # returning probability predictions, so only really need 1 'repetition' 
pred_method = 'probability' # return the model prediction as probability (for choice and PDW)
rt_method = 'mean'          # return the expected value of RT, given the model's prediction of the distribution

mods = np.array([1, 2, 3])
cohs = np.array([0.2, 0.7])
hdgs = np.array(np.linspace(-12, 12, num_hdgs))  # to plot a smooth fit
deltas = np.array([-3, 0, 3]) # now include deltas
data_pred, ntrials = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=False)

model_data, _ = ddm_2d.generate_data(params=fitted_params, data=data_pred, accumulator=accum, 
                                     method=pred_method, rt_method=rt_method, return_wager=False)


# %% Plot zero conflict data (all modalities)

# actual means from the data
# by_conds = ['modality', 'coherence', 'heading', 'delta']
# df_means = behavior_means(data, by_conds=by_conds, long_format=True).pipe(replicate_ves)
# pred_means = behavior_means(model_data, by_conds=by_conds) # redundant if nreps = 1

# g_conds = ['modality', 'coherence', 'delta']
# p0 = [[0, 3], [0.1, 0, 3, 0.5], [0.1, 0, 3, 0.5]]
# fit_results = gauss_fit_hdg_group(data, p0=p0, y_vars=('choice','PDW', 'RT'), by_conds=g_conds)
# fit_df = fit_results_to_dataframe(fit_results, by_conds=g_conds).pipe(replicate_ves)

by_conds = ['modality', 'coherence', 'heading']
df_means = behavior_means(data_delta0, by_conds=by_conds, long_format=True).pipe(replicate_ves)
df_means = df_means[df_means['variable'] != 'correct']
df_means = df_means[df_means['variable'] != 'PDW']


model_data_delta0 = model_data.loc[model_data['delta']==0, :].pipe(replicate_ves)

g= plot_behavior_hdg(df_means, model_data_delta0, row='variable', col='coherence',
                      hue='modality', palette=sns.color_palette("Set2", 3))


# %% Plot combined condition (different conflicts)

# combined only, all cue conflicts
# data_deltas = data_melt.loc[data_melt['modality']==3, :]
# g = sns.FacetGrid(data_deltas, row='variable', col='coherence', hue='delta')
# g.map(errbar_plot, 'heading', 'mean', 'sem', linestyle='-', marker='.')
# %%


def result_to_json(result, json_filename):

    del result['fun']

    result['x'] = result['x'].tolist()
    result['x0'] = result['x0'].tolist()

    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)
