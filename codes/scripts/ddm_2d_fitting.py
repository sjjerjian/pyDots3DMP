# %% imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import json

from ddm_moi.Accumulator import AccumulatorModelMOI
from ddm_moi import ddm_2d

from behavior.preprocessing import dots3DMP_create_trial_list, data_cleanup, format_onetargconf
from behavior.descriptive import *

from pybads import BADS

# %%
def main():

    # SIMULATED DATA

    # nreps = 200
    # generate_sim_data(nreps=nreps)
    # with open(f"../../data/sim_behavior_202308_{nreps}reps.pkl", "rb") as file:
    #     data, sim_params = pickle.load(file)
    # data = pd.read_csv(f"../../data/ddm_2d_sim_{nreps}reps.csv")
    # data_delta0 = data.loc[(data['delta']==0) & data['bound_hit'], :] # bound hit and delta == 0

    # %% ----------------------------------------------------------------
    # REAL DATA
    filename = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230606.csv"
    data = data_cleanup(filename)
    data = format_onetargconf(data, remove_one_targ=True)
    data_delta0 = data.loc[data['delta']==0, :]  # fit zero conflict data only

    # %% ----------------------------------------------------------------
    # Run first optimization for choice and RT only
    
    accum=dict()
    accum['tvec'] = np.arange(0, 2, 0.01)
    accum['grid_vec'] = np.arange(-3, 0, 0.025)

    # initial parameter 'guesses'
    init_params = {
        'kmult': [0.3, 0.4],
        'bound': [0.6, 1, 1],
        'ndt': .15, 
        'sigma_ndt': 0.05,
    }
    
    lb = np.array([0.05, 0.4, 0.4, 0.4, 0.05, 0.0])
    ub = np.array([10, 5, 5, 5, 0.4, 0.1])
    
    plb = np.array([0.1, 0.5, 0.5, 0.5, 0.1, 0.03])
    pub = np.array([5, 3, 3, 3, 0.3, 0.06])
    
    fixed = np.array([0, 0, 0, 0, 0, 1])
    
    # bads_result = run_bads(init_params, data_delta0, accum, fixed=fixed, bounds=(lb, ub, plb, pub), outputs=['choice', 'RT'], llh_scaling=[1, 1])
    # result_to_json(bads_result, "../../data/lucio_bads_result_noPDW.json")
    
    # fitted_params = ddm_2d.set_params_dict_from_array(bads_result.x, init_params)
    # fitted_params['sigma_ndt'] = init_params['sigma_ndt']
        
    # for plotting arbitrary parameters, without a fit result
    # fitted_params = {
    #     'kmult': [0.3, 0.4],
    #     'bound': [0.6, 1, 1],
    #     'ndt': .15, 
    #     'sigma_ndt': 0.001,
    # }
    
    model_data = model_predictions(fitted_params, accum, num_hdgs=11, return_wager=False)
    model_data_delta0 = model_data.loc[model_data['delta']==0, :].pipe(replicate_ves)
        
    plot_results(data_delta0, model_data_delta0, hue='modality', return_wager=False)
    
    # now take the params and rerun for theta and alpha (with tighter bounds on everything else)
    # init_params2 = ddm_2d.set_params_dict_from_array(bads_result.x, init_params)
    # init_params2['sigma_ndt'] = init_params['sigma_ndt']  # fix back at original (since it will have changed under the hood)
    # init_params2['theta'] = [0.8, 0.8, 0.8]
    # init_params2['alpha'] = 0.05
    # TODO set bounds here

    # lb = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.1, 0.1, 0.1, 0])
    # ub = np.array([2, 2, 1, 1, 2, 2, 0.4, 0.4, 0.4, 0.15])

    # plb = np.array([0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.15, 0.15, 0.15, 0.03])
    # pub = np.array([1,  1, 0.7, 0.7, 1.5, 1.5, 0.35, 0.35, 0.35, 0.1])

    # # bads_result2 = run_bads(init_params2, accum, data_delta0, bounds=(lb, ub, plb, pub), ['choice', 'PDW', 'RT'], [1, 1, 0.1]])



# %% ----------------------------------------------------------------

def run_bads(init_params, data, accum_kw, fixed=None, bounds=None, outputs=['choice', 'RT'], llh_scaling=[1, 1]):
    
    init_params_array = ddm_2d.get_params_array_from_dict(init_params)
    
    if fixed is None:
        fixed = np.zeros_like(init_params_array)

    if bounds is None:
        # set bounds and plausible bounds
        lb, ub = init_params_array * 0.1, init_params_array * 2
        plb, pub = init_params_array * 0.3, init_params_array * 1.5
    else:
        lb, ub, plb, pub = bounds

    # TODO eventually initialize multiple times from different starting points, loop over function, add decorator?

    # lambda function to handle custom arguments
    target = lambda params: ddm_2d.objective(params, init_params, fixed,
                                            data=data, accum_kw=accum_kw,
                                            stim_scaling=True,
                                            outputs=outputs, llh_scaling=llh_scaling)

    bads = BADS(target, init_params_array, lb, ub, plb, pub)
    bads_result = bads.optimize()  
    
    return bads_result


# %% ----------------------------------------------------------------
# ===== generate model predictions
# the optimization returned just the loss function values, now we can rerun the generate_data part while holding model parameters fixed at their fitted values
# to generate model predicted data points for each condition (which doesn't have to be the same as the ones we fit)
# e.g. we can predict for a linearly spaced set of headings to get a smooth curve, and we can predict for cue conflict conditions,
# though we only fit to zero conflict trials

def model_predictions(fitted_params, accum, num_hdgs=33, return_wager=True):

    # to plot a smooth curve of model prediction over headings
    max_hdg = 12
    hdgs = np.array(np.linspace(-max_hdg, max_hdg, num_hdgs))
            
    pred_method = 'probability' # return the model prediction as probability (for choice and PDW)
    nreps = 1                   # returning probabilistic predictions, so only really need 1 'repetition' 
    rt_method = 'mean'          # return the expected value of RT, given the model's prediction of the distribution

    # rest of conditions list
    mods = np.array([1, 2, 3])
    cohs = np.array([0.2, 0.7])
    # deltas = np.array([-3, 0, 3]) # now include deltas
    deltas = np.array([0]) # zero delta only, for quicker testing

    data_pred, ntrials = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=False)
    model_data, _ = ddm_2d.generate_data(params=fitted_params, data=data_pred, accum_kw=accum, 
                                        method=pred_method, rt_method=rt_method, stim_scaling=True, return_wager=return_wager)
    
    return model_data


# %% ----------------------------------------------------------------
# Plot zero conflict data and model predictions (all modalities)

def plot_results(emp_data, model_data, hue, return_wager=True):

    # actual means from the data
    df_means = behavior_means(emp_data, by_conds=[hue, 'coherence', 'heading'], long_format=True).pipe(replicate_ves)
    df_means = df_means[df_means['variable'] != 'correct']
    
    if not return_wager:
        df_means = df_means[df_means['variable'] != 'PDW']

    g = plot_behavior_hdg(df_means, model_data, row='variable', col='coherence',
                          hue=hue, palette=sns.color_palette("Set2", 3))
    
    return g

# ----------------------------------------------------------------
# helper function to save bads_results to a json file
def result_to_json(result, json_filename):

    del result['fun']
    result['x'] = result['x'].tolist()
    result['x0'] = result['x0'].tolist()

    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)

# %% ----------------------------------------------------------------
# ===== generate (and save) simulated data =====

def generate_sim_data(nreps=200):
    
    sim_params = {
        'kmult': 0.25,
        'bound': np.array([0.8, 0.8]),
        'ndt': [0.2, 0.3, 0.25], 
        'sigma_ndt': 0.05,
        'sigma_dv': 1,
        'alpha': 0.03, 
        'theta': [1.2, 1, 1.1],
    }
            
    mods = np.array([1, 2, 3])
    cohs = np.array([0.2, 0.7])
    hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
    deltas = np.array([-3, 0, 3])

    trial_table, ntrials = dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=False)

    accum = AccumulatorModelMOI(tvec=np.arange(0, 2, 0.005), grid_vec=np.arange(-3, 0, 0.025))

    sim_method = 'sim'
    save_dv = False
    sim_data, log_odds_maps = ddm_2d.generate_data(sim_params, data=trial_table, 
                                                   accumulator=accum, method=sim_method,
                                                   save_dv=save_dv, return_wager=True)

    if save_dv:
        with open(f"../../data/ddm_2d_{sim_method}_{nreps}reps.pkl", "wb") as file:
            pickle.dump((sim_data, sim_params), file, protocol=-1)
    else:
        sim_data.to_csv(f"../../data/ddm_2d_{sim_method}_{nreps}reps.csv")
    
    return sim_data, sim_params, log_odds_maps


if __name__ == "__main__":
    main()