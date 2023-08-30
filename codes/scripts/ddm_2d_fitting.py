# %% imports

import numpy as np
import pickle

from ddm_moi import Accumulator
from ddm_moi.ddm_2d import *

from behavior.preprocessing import dots3DMP_create_trial_list
from behavior.descriptive import behavior_means, plot_behavior_hdg

from pybads import BADS
#from scipy.optimize import minimize

# %% generate (and save) simulated data

sim_params = {'kmult': 15, 'bound': np.array([1, 1]), 'alpha': 0.05, 'theta': [0.8, 0.6, 0.7],
                'ndt': [0.1, 0.3, 0.2], 'sigma_ndt': 0.06, 'sigma_dv': 1}
        
mods = np.array([1, 2, 3])
cohs = np.array([0.3, 0.7])
hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
deltas = np.array([-3, 0, 3])

nreps = 200
trial_table, ntrials = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=False)

accum = Accumulator.AccumulatorModelMOI(tvec=np.arange(0, 2, 0.005), grid_vec=np.arange(-3, 0, 0.025))
sim_data, _ = ddm_2d_generate_data(sim_params, data=trial_table, 
                                         accumulator=accum, method='simulate', return_wager=True)

with open(f"../data/sim_behavior_202308_{nreps}reps.pkl", "wb") as file:
    pickle.dump((sim_data, sim_params), file, protocol=-1)


# %% load data and initialize objective function

filename = f"../data/sim_behavior_202308_{nreps}reps.pkl" # replace with real data

with open(f"../data/sim_behavior_202308_{nreps}reps.pkl", "rb") as file:
    data, sim_params = pickle.load(file)

# initialize accumulator
accum = AccumulatorModelMOI(tvec=np.arange(0, 2, 0.05), grid_vec=np.arange(-3, 0, 0.025))

# TODO should be able to set init_params as regular dict, and order it 
# set initial parameters for optimization

# first fitting choice and RT only
init_params = {'kmult': 0.3, 'bound': np.array([0.5, 0.5]), 'ndt': [0.3, 0.3, 0.3], 'sigma_ndt': 0.06}
init_params_array = get_params_array_from_dict(init_params)

fixed = np.ones_like(init_params_array)
fixed[0:3] = 0  # fitting kmult and bound only
#fixed[:-1] = 0 # fitting all except sigma_ndt

# set bounds and plausible bounds, for BADS

# shortcut
lb, ub = init_params_array * 0.1, init_params_array * 5
plb, pub = init_params_array * 0.3, init_params_array * 3

# lb = np.array([0.1, 0.95, 0.95, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0])
# ub = np.array([0.2, 1.05, 1.05, 0.15, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.2])

# plb = np.array([0.14, 0.98, 0.98, 0.03, 0.6, 0.6, 0.6, 0.05, 0.1, 0.1, 0.03])
# pub = np.array([0.16, 1.02, 1.02, 0.1, 1, 1, 1, 0.35, 0.35, 0.35, 0.1])

# first fit choice and RT only
target = lambda params: ddm_2d_objective(params, init_params, fixed,
                                         data=data, accumulator=accum,
                                         outputs=['choice', 'RT'], llh_scaling=[1, 0.1])

# fit using Bayesian Adaptive Direct Search (BADS)
fit_options = {'random_seed': 42} # for reproducibility
bads = BADS(target, init_params_array, lb, ub, plb, pub, options=fit_options)

# %% ===== Fit choice and RT =====

bads_result = bads.optimize()  # not bad...

# using scipy optimize minimize
#args_tuple = (init_params, fixed, param_keys, data, accum, ['choice', 'RT'], [1, 0.1])
#fit_options = {'disp': True, 'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True}

# res = minimize(ddm_2d_objective, init_params_array, method='Nelder-Mead',
#                args=args_tuple, options=fit_options)

# %% re-run optimization, but now with PDW. set tighter bounds on parameters already fitted

# TODO rerun second fit fitting all three (with tight bounds on all parameters except alpha and theta)
# TODO rerun objective with all values fixed, to return model_data
# TODO plot fit results - 'rerun' objective with params held fixed and a continuous range of headings


# %% now take the fitted parameters, and generate predicted data for the same conditions as observed data

fitted_params = {
    'kmult': 0.3, 'bound': np.array([0.5, 0.5]), 'ndt': [0.3, 0.3, 0.3], 'sigma_ndt': 0.06,
    'alpha': 0.05, 'theta': [0.8, 0.6, 0.7], 
}

# fitted_params = OrderedDict([
#     ('kmult', 0.1264528),
#     ('bound', np.array([1.22985072, 1.24437893])),
#     ('alpha', 0.1),
#     ('theta', [0.8, 0.6, 0.7]),
#     ('ndt', [0.03090672, 0.21717266, 0.10039247]),
#     ('sigma_ndt', 0.06),
# ])

num_hdgs = 200              # to plot a smooth curve of model prediction over headings
nreps = 1                   # returning probability predictions, so only really need 1 'repetition' 
pred_method = 'probability' # return the model prediction as probability (for choice and PDW)
rt_method = 'mean'          # return the expected value of RT, given the model's prediction of the distribution
# alternative here would be to use 'sample' or 'simulate' methods to generate actual trials, then calculate averages

mods = np.array([1, 2, 3])
cohs = np.array([0.3, 0.7])
hdgs = np.array(np.linspace(-12, 12, num_hdgs))  # to plot a smooth fit
deltas = np.array([-3, 0, 3])
data_pred, ntrials = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=False)

model_data, _ = ddm_2d_generate_data(params=fitted_params, data=data_pred,
                                     accumulator=accum, 
                                     method=pred_method rt_method=rt_method,
                                     return_wager=True,
                                     )

# actual means from the data
by_conds = ['modality', 'coherence', 'heading', 'delta']
data_means = behavior_means(data, by_conds=by_conds)
pred_means = behavior_means(data_pred, by_conds=by_conds) # technically redundant if nreps = 1

# TODO plot it






print('Main done')