
"""
Example script to run BADS optimization on 2-D accumulator model parameters, using data
from dots3DMP task.

Steven Jerjian 2024-01-20

"""

# %% ===== IMPORTS =====

import numpy as np

import pickle
import json
from datetime import datetime

from pybads import BADS

from codes.ddm_moi import ddm_2d

# going to use these custom functions for processing behavioral data, and plotting fit/data results
from codes.behavior.preprocessing import (
    dots3DMP_create_trial_list, dots3DMP_create_conditions,
    data_cleanup, format_onetargconf,
    )
from codes.behavior.descriptive import plot_behavior_hdg, behavior_means, replicate_ves


# %% ===== LOAD DATA  ======

# - simulated data, serialized with simulation parameters
# - see generate_sim_data helper function further down
# datafilepath = "../data/ddm_2d_sample_100reps.pkl";
# with open(datafilepath, "rb") as file:
#     data, sim_params, log_odds_maps = pickle.load(file)

# - empirical data, saved as csv file
# (using struct2csv in matlab to save out data struct as csv file, with fields as column names)
datafilepath = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230606.csv"
data = data_cleanup(datafilepath)  # this is a hack function to quickly load and clean the data, could be improved/generalized with options
data = format_onetargconf(data, remove_one_targ=True)

# %% ===== SET UP KEY VARIABLES ======

# use zero conflict data only for the fitting part
data_delta0 = data.loc[data['delta'] == 0, :]

# set up the conditions table for the prediction part (this will come later)
hdgs = np.unique(data['heading'])
conds = {'mods': [1, 2, 3],
        'cohs': [0.3, 0.7],
        'deltas': [-3, 0, 3],
        'hdgs': hdgs}
pred_conds = dots3DMP_create_conditions(conds, cond_labels=['modality', 'coherence', 'delta', 'heading'])

# initialize accumulator variables via dict
accum_kw = {'tvec': np.arange(0, 2, 0.005), 'grid_vec': np.arange(-3, 0, 0.025)}

# pre-define stimulus urgency time-course, since it will be fixed for all fit iterations
stim_sc = ddm_2d.get_stim_urgs(tvec=accum_kw['tvec'])

# starting parameters
init_params = {
    'kmult': [0.4, 0.3, 0.6],
    'bound': [0.6, 0.7, 0.7],
    'ndt': [0.2, 0.3, 0.25],
    'sigma_ndt': 0.05,
    'theta': [1.05, 0.95, 1.0],  # in units of log odds
    'alpha': 0.05,
}

# first argument of BADS requires an array of inputs
init_params_array = ddm_2d.get_params_array_from_dict(init_params)
fixed = np.zeros_like(init_params_array)

outputs = ['choice', 'PDW', 'RT'] # what do we want to include in likelihood calculation
out_scaling = [1, 1, 1]           # scale the likelihoods of each output by this [1,1,1] is the default

# bounds and plausible bounds for BADS
lb, ub = init_params_array * 0.1, init_params_array * 2
plb, pub = init_params_array * 0.3, init_params_array * 1.5

# %% ===== RUN BADS OPTIMIZATION =====

# TODO initialize BADS multiple times from different starting points

# use an anonymous function to pass extra arguments to the objective function
target = lambda params: ddm_2d.objective(params, init_params, fixed,
                                        data=data_delta0, accum_kw=accum_kw,
                                        stim_scaling=stim_sc, method='prob',
                                        outputs=outputs, llh_scaling=out_scaling)

# BADS class takes in the target objective function, array of initial parameters, and bounds
bads = BADS(target, init_params_array, lb, ub, plb, pub)
bads_result = bads.optimize()

save_file_name = f"../data/{subject}_bads_result_{datetime.now.strftime('%Y%m%d_%H%M%S')}.json"
result_to_json(bads_result, save_file_name)


# %% ===== GENERATE MODEL PREDICTIONS =====

# NOTE There are 2 steps here (although perhaps there's a cleaner way to avoid step 1)

# 1. We first re-call the objective function with the final fitted parameters and "training" data
# to get the log_odds_maps (since we don't get it out when we run the objective function within
# the optimization routine

# reset the fixed parameters to their initial set values
# and make a dictionary of the fitted params array to pass to
init_params_array, fitted_params_array = bads_result.x0, bads_result.x
fitted_params_array[fixed == 1] = init_params_array[fixed == 1]
fitted_params = ddm_2d.set_params_dict_from_array(fitted_params_array, init_params)

# now we can use get_llhs, which runs the objective function without applying the fit wrapper
# NOTE this is what you would use to "hand-check" init_params, without running optimization
# just substitute init_params in for the params here
neg_llh, model_llh, model_data, log_odds_maps = \
    ddm_2d.get_llhs(params=fitted_params, data=data,
                    outputs=outputs, accum_kw=accum,
                    method='prob', stim_scaling=stim_scaling, llh_scaling=llh_scaling)

# 2. NOW we can call generate_data explicitly with the final fitted parameters and a set of conditions
# for predictions. This set of conditions can be different to the original fit (e.g. we only
# fit the model to zero conflict trials, but can make predictions on )
# pred_conds then specifies the trial conditions we want to just get a prediction for

# note that rt_method is now set to "mean", or "peak" so that we return an actual RT prediction
model_data, _, _ = \
    ddm_2d.generate_data(params=fitted_params, data=pred_conds,
                        accum_kw=accum, method='prob', rt_method='mean',
                        stim_scaling=stim_scaling, return_wager=return_wager,
                        log_odds_maps=log_odds_maps)


# %% ===== PLOT RESULTS =====

show_PDW = True

# first for zero conflict
model_data_delta0 = model_data.loc[model_data['delta'] == 0, :].pipe(replicate_ves)
plot_results(data_delta0, model_data_delta0, hue='modality', return_wager=show_PDW)

# and now for cue conflict
data_deltas = data.loc[data['modality'] == 3, :]
model_deltas = model_data.loc[model_data['modality'] == 3, :]
plot_results(data_deltas, model_deltas, hue='delta', return_wager=show_PDW)




# END OF MAIN CODE
# ----------------------------------------------------------------



# %% HELPER FUNCTIONS


def generate_sim_data(sim_params: dict, conds_dict: dict, sim_method='sample', nreps=200, save_sim_params=False):
    """Generate simulated data with known parameters using 2-D accumulator model in order to test model recovery.

    Args:
        sim_params (dict): dictionary of simulation parameters
        conds_dict (dict): dictionary of conditions (for input to dots3DMP_create_trial_list)
        sim_method (str, optional): Method for simulation, either 'sample' or 'sim_dv'. Defaults to 'sample'.
        nreps (int, optional): number of reps of each condition. Defaults to 200.
        save_sim_params (bool, optional): False will save just the data results as csv.
        True will pickle data and the used simulation parameters. Defaults to False.

    Returns:
        _type_: _description_
    """

    drop_no_hit = False # drop trials that fail to hit bound
    return_wager = True # return PDW? setting this to false will skip PDF calculations
    save_dv = False     # save the individual trial dvs? only relevant for sim_method = 'sim"


    trial_table = dots3DMP_create_trial_list(**conds_dict, nreps=nreps, shuff=False)
    accum = {'tvec': np.arange(0, 2, 0.01), 'grid_vec': np.arange(-3, 0, 0.025)}

    # return wager and use stim scaling by default

    sim_data, log_odds_maps, full_dvs = ddm_2d.generate_data(sim_params, data=trial_table, accum_kw=accum,
                                                             method=sim_method,
                                                             stim_scaling=True,
                                                             return_wager=return_wager,
                                                             save_dv=False)

    if drop_no_hit:
        keep_trials = sim_data['bound_hit']
        sim_data = sim_data.loc[keep_trials, :]
        if save_dv:
            full_dvs = full_dvs[:, :, keep_trials]

    df_means = behavior_means(sim_data,
                              by_conds=['modality', 'coherence', 'heading', 'delta'],
                              drop_na=False,
                              long_format=True).pipe(replicate_ves)

    if not return_wager:
        df_means = df_means.loc[df_means['variable'] != 'PDW', :]

    # separate plots for zero delta and cue conflict
    df_means_delta0 = df_means.loc[df_means['delta'] == 0, :]
    df_means_deltas = df_means.loc[df_means['modality'] == 3, :]

    palette = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    plot_behavior_hdg(df_means_delta0, row='variable', col='coherence', hue='modality', palette=palette)

    palette = [(0, 0, 1), (0, 1, 1), (0, 1, 0)]
    plot_behavior_hdg(df_means_deltas, row='variable', col='coherence', hue='delta', palette=palette)

    cur_time_str = datetime.now.strftime("%Y%m%d_%H%M%S")
    if save_sim_params:
        with open(f"../data/ddm_2d_{sim_method}_{nreps}reps_{cur_time_str}.pkl", "wb") as file:
            pickle.dump((sim_data, sim_params, log_odds_maps), file, protocol=-1)
    else:
        sim_data.to_csv(f"../data/ddm_2d_{sim_method}_{nreps}reps_{cur_time_str}.csv")

    return sim_data, log_odds_maps, full_dvs, sim_params


def result_to_json(result, json_filename):
    """Helper function to save bads_results to a json file."""

    del result['fun']  # can't be json-ified, get rid of it
    result['x'] = result['x'].tolist() # convert np.array to list
    result['x0'] = result['x0'].tolist()

    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)



# helper function to plot results of model predictions
def plot_results(emp_data, model_data, hue, return_wager=True):
    """Helper function to plot results of model predictions."""

    # actual means from the data
    df_means = behavior_means(emp_data, by_conds=[hue, 'coherence', 'heading'], long_format=True)

    df_means = df_means[df_means['variable'] != 'correct']

    if not return_wager:
        df_means = df_means[df_means['variable'] != 'PDW']

    if hue == 'modality':
        df_means = replicate_ves(df_means)
        palette = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    elif hue == 'delta':
        palette = [(0, 1, 1), (0, 0, 1), (0, 1, 0)]

    g = plot_behavior_hdg(df_means, model_data, row='variable', col='coherence',
                          hue=hue, palette=palette)
    # TODO add legend

    return g


# ----------------------------------------------------------------

"""
# example of how to generate simulated data with a given set of parameters

sim_params = {
    'kmult': [0.35, 0.35, 1.4],
    'bound': [0.6, 0.75, 1.5],
    'ndt': 0.25,
    'sigma_ndt': 0.05,
    'sigma_dv': 1,             # this is only relevant for actual dv simulation
    'theta': [1.2, 0.9, 1.1],  # in units of log odds
    'alpha': 0.05,
}

# define unique conditions
conds = {'mods': [1, 2, 3],
        'cohs': [0.3, 0.7],
        'deltas': [-3, 0, 3],
        'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]}

sim_data, log_odds_maps, full_dvs, sim_params = generate_sim_data(
    conds_dict=conds,
    sim_params=sim_params,
    sim_method="sample",            # draw samples from model posteriors
    nreps=200, save_sim_params=True)

"""
