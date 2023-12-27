import numpy as np
import pandas as pd

import pickle
import json
from typing import Optional

from ddm_moi import ddm_2d

from behavior.preprocessing import (
    dots3DMP_create_trial_list, dots3DMP_create_conditions,
    data_cleanup, format_onetargconf,
    )

from behavior.descriptive import (
    plot_behavior_hdg, behavior_means, replicate_ves
    )

from pybads import BADS

# %%



def main(init_params: dict, bounds: Optional[tuple[np.ndarray]] = None,
         fixed: Optional[np.ndarray] = None,
         subject='sim', run_fit=False, **output_kwargs):

    if subject == 'sim':
        with open("../data/ddm_2d_sample_100reps.pkl", "rb") as file:
            data, sim_params, log_odds_maps = pickle.load(file)
    elif subject == 'lucio':
        filename = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230606.csv"
        data = data_cleanup(filename)
        data = format_onetargconf(data, remove_one_targ=True)

    # fit zero conflict data only
    data_delta0 = data.loc[data['delta'] == 0, :]

    accum = {'tvec': np.arange(0, 2, 0.01), 'grid_vec': np.arange(-3, 0, 0.025)}

    stim_sc = ddm_2d.get_stim_urgs(tvec=accum['tvec'])

    # max_hdg = 12
    # hdgs = np.array(np.linspace(-max_hdg, max_hdg, 33))
    hdgs = np.unique(data_delta0['heading'])

    conds = {'mods': [1, 2, 3],
             'cohs': [0.3, 0.7],  # 0.3, 0.7 for simulation
             'deltas': [-3, 0, 3],
             'hdgs': hdgs}

    pred_conds = dots3DMP_create_conditions(conds, ['modality', 'coherence', 'delta', 'heading'])


    fitted_params = init_params
    if run_fit:

        bads_result = run_bads(init_params, data_delta0, accum, bounds=bounds, fixed=fixed,
                               stim_scaling=stim_sc, **output_kwargs)
        #  TODO add unique identified here
        result_to_json(bads_result, f"../data/{subject}_bads_result.json")

        # init_params_array = ddm_2d.get_params_array_from_dict(init_params)
        init_params_array, fitted_params_array = bads_result.x0, bads_result.x
        fitted_params_array[fixed == 1] = init_params_array[fixed == 1]

        fitted_params = ddm_2d.set_params_dict_from_array(fitted_params_array, init_params)

    # generate model predictions, based on fitted parameters
    # actually have to pass in fitting data again, to get the log_odds_maps
    # pred_conds then specifies the trial conditions we want to just get a prediction for

    return_wager = True
    if 'outputs' in output_kwargs:
        return_wager = 'PDW' in output_kwargs['outputs']

    model_data, neg_llh, model_llh = model_predictions(
        fitted_params, data_delta0, pred_conds=pred_conds,
        accum=accum, stim_scaling=stim_sc, return_wager=return_wager)

    model_data_delta0 = model_data.loc[model_data['delta'] == 0, :].pipe(replicate_ves)

    # plot_results(data_delta0, model_data_delta0, hue='modality', return_wager=return_wager)

    data_deltas = data.loc[data['modality'] == 3, :]
    model_deltas = model_data.loc[model_data['modality'] == 3, :]

    plot_results(data_deltas, model_deltas, hue='delta', return_wager=return_wager)


# %% ----------------------------------------------------------------

def run_bads(init_params: dict, data: pd.DataFrame, accum_kw: dict, bounds=None, fixed=None,
             stim_scaling=True, outputs=['choice', 'PDW', 'RT'], llh_scaling=None):

    init_params_array = ddm_2d.get_params_array_from_dict(init_params)

    if fixed is None:
        fixed = np.zeros_like(init_params_array)

    if bounds is None:
        lb, ub = init_params_array * 0.1, init_params_array * 2
        plb, pub = init_params_array * 0.3, init_params_array * 1.5
    else:
        lb, ub, plb, pub = bounds

    # TODO eventually initialize multiple times from different starting points

    # lambda function to handle extra arguments
    target = lambda params: ddm_2d.objective(params, init_params, fixed,
                                             data=data, accum_kw=accum_kw,
                                             stim_scaling=stim_scaling, method='prob',
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

def model_predictions(fitted_params: dict, fit_data: pd.DataFrame, pred_conds=None, accum=None,
                      stim_scaling=True, outputs=None, llh_scaling=None, return_wager=True):

    if accum is None:
        accum = {'tvec': np.arange(0, 2, 0.01),  # 0.005
                 'grid_vec': np.arange(-3, 0, 0.025),  # 0.025
                 }

    if outputs is None:
        outputs = ['choice', 'PDW', 'RT']

    if llh_scaling is None:
        llh_scaling = np.ones(len(outputs))

    # don't need to pass in return_wager here, because the objective function
    # will already check whether 'PDW' is in outputs

    # rt_method defaults to 'likelihood'
    neg_llh, model_llh, model_data, log_odds_maps = \
        ddm_2d.get_llhs(params=fitted_params, data=fit_data,
                        outputs=outputs, accum_kw=accum,
                        method='prob', stim_scaling=stim_scaling, llh_scaling=llh_scaling)

    # generate model predictions using final fitted parameters (can be on
    # another set of conditions (e.g. finely spaced headings, deltas),
    # but we have to use the pre-calculated log odds maps!)
    # rt_method is now set to "mean", or "peak"

    model_data, _, _ = \
        ddm_2d.generate_data(params=fitted_params, data=pred_conds,
                             accum_kw=accum, method='prob', rt_method='mean',
                             stim_scaling=stim_scaling, return_wager=return_wager,
                             log_odds_maps=log_odds_maps)

    return model_data, neg_llh, model_llh


# %% ----------------------------------------------------------------
# Plot zero conflict data and model predictions (all modalities)

def plot_results(emp_data, model_data, hue, return_wager=True):

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


# %% ----------------------------------------------------------------
# helper function to save bads_results to a json file
def result_to_json(result, json_filename):

    del result['fun']  # can't be json-ified
    result['x'] = result['x'].tolist()
    result['x0'] = result['x0'].tolist()

    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)


# %% ----------------------------------------------------------------

def loss_landscape(param_key, param_index=None, num_step=10, use_sim_data=True):

    if use_sim_data:
        with open("../data/ddm_2d_samp_300reps.pkl", "rb") as file:
            data, sim_params, log_odds_maps = pickle.load(file)

    else:
        filename = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230606.csv"
        data = data_cleanup(filename)
        data = format_onetargconf(data, remove_one_targ=True)

    # fit zero conflict data only
    data_delta0 = data.loc[data['delta'] == 0, :]

    accum = {'tvec': np.arange(0, 2, 0.005),  # 0.005
             'grid_vec': np.arange(-3, 0, 0.025),  # 0.025
             }

    stim_sc = ddm_2d.get_stim_urgs(tvec=accum['tvec'])

    params = {
        'kmult': [0.1, 0.1, 0.3],
        'bound': [0.6, 0.7, 0.7],
        'ndt': [0.2, 0.3, 0.2],
        'sigma_ndt': 0.05,
        'theta': [1.1, 0.8, 0.9],  # in units of log odds
        'alpha': 0.03,
    }

    if param_index is None:
        base_val = params[param_key]
    else:
        base_val = params[param_key][param_index]

    param_range = base_val*0.5, base_val*5
    param_vals = np.linspace(param_range[0], param_range[1], num_step)

    neg_llhs, model_llhs = [], []

    for kval in param_vals:

        if param_index is None:
            params[param_key] = kval
        else:
            params[param_key][param_index] = kval

        for key, val in params.items():
            print(f"{key}: {val}\t")

        _, neg_llh, model_llh = model_predictions(
            params, data_delta0, accum=accum,
            stim_scaling=stim_sc, return_wager=False
            )

        neg_llhs.append(neg_llh)
        model_llhs.append(model_llh)

    return neg_llhs, model_llhs, param_vals


# %% ----------------------------------------------------------------
# ===== generate (and save) simulated data =====

def generate_sim_data(sim_params: dict, sim_method='sample', nreps=200, full_save=False):

    # TODO fix random calls in sim rvs (allow seed or random)

    drop_no_hit = False

    conds = {'mods': [1, 2, 3],
             'cohs': [0.3, 0.7],
             'deltas': [-3, 0, 3],
             'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]}

    trial_table = dots3DMP_create_trial_list(**conds, nreps=nreps, shuff=False)

    accum = {'tvec': np.arange(0, 2, 0.01),  # 0.005
             'grid_vec': np.arange(-3, 0, 0.025), # 0.025
             }

    # return wager and use stim scaling by default
    save_dv = False
    return_wager = False
    sim_data, log_odds_maps, full_dvs = ddm_2d.generate_data(sim_params, data=trial_table, accum_kw=accum,
                                                             method=sim_method,
                                                             stim_scaling=True,
                                                             return_wager=return_wager,
                                                             save_dv=save_dv)

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

    df_means_delta0 = df_means.loc[df_means['delta'] == 0, :]
    df_means_deltas = df_means.loc[df_means['modality'] == 3, :]

    palette = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    plot_behavior_hdg(df_means_delta0, row='variable', col='coherence', hue='modality',
                      palette=palette)

    palette = [(0, 0, 1), (0, 1, 1), (0, 1, 0)]
    plot_behavior_hdg(df_means_deltas, row='variable', col='coherence', hue='delta',
                      palette=palette)

    if full_save:
        with open(f"../data/ddm_2d_{sim_method}_{nreps}reps.pkl", "wb") as file:
            pickle.dump((sim_data, sim_params, log_odds_maps), file, protocol=-1)
    else:
        sim_data.to_csv(f"../data/ddm_2d_{sim_method}_{nreps}reps.csv")

    return sim_data, log_odds_maps, full_dvs, sim_params


# %%
if __name__ == "__main__":

    sim_params = {
        'kmult': [0.35, 0.35, 1.4],
        'bound': [0.6, 0.75, 1.5],
        'ndt': 0.25,
        'sigma_ndt': 0.05,
        'sigma_dv': 1,
        'theta': [1.2, 0.9, 1.1],  # in units of log odds
        'alpha': 0.05,
    }

    sim_data, log_odds_maps, full_dvs, sim_params = generate_sim_data(
        sim_params=sim_params, nreps=200, full_save=True)


    # simulation fitting

    # initial parameter 'guesses' first fit attempt
    # init_params = {
    #     'kmult': [0.15, 0.4],
    #     'bound': [0.5, 0.5, 0.5],
    #     'ndt': 0.25,
    #     'sigma_ndt': 0.05,
    #     'theta': [1, 1, 1],  # in units of log odds
    #     'alpha': 0.05,
    # }

    # init_params = {
    #     'kmult': [0.06776596, 0.25264891],
    #     'bound': [0.62440945, 0.65017875, 0.71219019],
    #     'ndt': 0.25,
    #     'sigma_ndt': 0.05,
    #     'theta': [1.21665203, 0.94460407, 1.26609347],  # in units of log odds
    #     'alpha': 0.05262214479711564,
    # }

    # # # initial settings, see what the error is
    # init_params = {
    #     'kmult': [0.7, 2],
    #     'bound': [0.6, 0.75, 0.7],
    #     'ndt': 0.25,
    #     'sigma_ndt': 0.05,
    #     'theta': [1.2, 0.9, 1.1],  # in units of log odds
    #     'alpha': 0.05,
    # }

    # lb = np.array([0.05, 0.05, 0.3, 0.3, 0.3, 0.1, 0.0, 0.5, 0.5, 0.5, 0.0])
    # ub = np.array([0.3, 0.55, 0.9, 0.9, 0.9, 0.4, 0.1, 1.5, 1.5, 1.5, 0.1])

    # plb = np.array([0.08, 0.1, 0.4, 0.4, 0.4, 0.2, 0.03, 0.6, 0.6, 0.6, 0.03])
    # pub = np.array([0.2, 0.45,  0.8, 0.8, 0.8, 0.3, 0.08, 1.3, 1.3, 1.3, 0.08])

    # fixed = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    # main(init_params, subject='sim', run_fit=False)

    # main(init_params, bounds=(lb, ub, plb, pub), fixed=fixed, subject='sim', run_fit=True, outputs=['choice', 'PDW'])


    # %% LUCIO, choice+RT only

    # init_params = {
    #     'kmult': [0.6, 0.4, 0.9],
    #     'bound': [0.55, 0.7, 0.8],
    #     'ndt': [0.28, 0.31, 0.25],
    #     'sigma_ndt': 0.05,
    # }

    # lb = np.array([0.5, .25, 0.75, .45, 0.6, 0.6, 0.2, 0.2, 0.2, 0.0])
    # ub = np.array([0.7, .55, 1.2, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.1])

    # plb = np.array([.55, 0.3, 0.8, 0.5, .65, .65, .24, .24, .24, .03])
    # pub = np.array([.65, 0.5, 1.15, .75, 0.9, 0.9, .35, .35, .34, .08])

    # fixed = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    # main(init_params, subject='lucio', run_fit=False)

    # main(init_params, bounds=(lb, ub, plb, pub), fixed=fixed, subject='lucio',
    #      run_fit=True, outputs=['choice', 'RT'], llh_scaling=[1, 0.5])


    # with PDW

    # init_params = {
    #     'kmult': [0.4, 0.3, 0.6],
    #     'bound': [0.6, 0.7, 0.7],
    #     'ndt': [0.2, 0.3, 0.25],
    #     'sigma_ndt': 0.05,
    #     'theta': [1.05, 0.95, 1.0],  # in units of log odds
    #     'alpha': 0.05,
    # }

    # lb = np.array([0.5, .25, 1.0, .45, 0.6, 0.6, 0.2, 0.2, 0.2, 0.0, 0.8, 0.8, 0.8, 0.0])
    # ub = np.array([0.7, .55, 1.2, .75, 0.9, 0.9, 0.4, 0.4, 0.4, 0.1, 1.3, 1.3, 1.3, 0.1])

    # plb = np.array([.55, 0.3, 1.05, 0.5, .65, .65, .24, .24, .24, .03, 0.9, 0.9, 0.9, .02])
    # pub = np.array([.65, 0.5, 1.15, 1.0, 1.0, 1.0, .35, .35, .34, .08, 1.2, 1.2, 1.2, .08])

    # fixed = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1])

    # main(init_params, subject='lucio', run_fit=False)

    # main(init_params, bounds=(lb, ub, plb, pub), fixed=fixed, subject='lucio',
    #       run_fit=True, outputs=['choice', 'PDW'], llh_scaling=[1, 1])




# testing loss landscape, clean this up
    # neg_llh0, model_llh0, param_val0 = loss_landscape('ndt', param_index=0, use_sim_data=True)
    # neg_llh1, model_llh1, param_val1 = loss_landscape('ndt', param_index=1, use_sim_data=True)
    # neg_llh2, model_llh2, param_val2 = loss_landscape('ndt', param_index=2, use_sim_data=True)

    # choice_llh0 = -np.array([d['choice'] for d in model_llh0])
    # rt_llh0 =  -np.array([d['RT'] for d in model_llh0])

    # choice_llh1 = -np.array([d['choice'] for d in model_llh1])
    # rt_llh1 =  -np.array([d['RT'] for d in model_llh1])

    # choice_llh2 = -np.array([d['choice'] for d in model_llh2])
    # rt_llh2 =  -np.array([d['RT'] for d in model_llh2])

    # # fig, ax = plt.subplots(3, 2)
    # # ax[0, 0].plot(param_val0, neg_llh0, label='kves')
    # # ax[0, 0].plot(param_val1, neg_llh1, label='kvis_lo')
    # # ax[0, 1].plot(param_val2, neg_llh2, label='kvis_hi')
    # # ax[0, 0].set_ylabel('Total llh')

    # # ax[1, 0].plot(param_val0, choice_llh0, label='kves')
    # # ax[1, 0].plot(param_val1, choice_llh1, label='kvis_lo')
    # # ax[1, 1].plot(param_val2, choice_llh2, label='kvis_hi')
    # # ax[1, 0].set_ylabel('Choice llh')

    # # ax[2, 0].plot(param_val0, rt_llh0, label='kves')
    # # ax[2, 0].plot(param_val1, rt_llh1, label='kvis_lo')
    # # ax[2, 1].plot(param_val2, rt_llh2, label='kvis_hi')
    # # ax[2, 0].set_ylabel('RT llh')


    # fig, ax = plt.subplots(3, 1, sharex=True)
    # ax[0].plot(param_val0, neg_llh0, label='ves', color='k', marker='.')
    # ax[0].plot(param_val1, neg_llh1, label='vis', color='r', marker='.')
    # ax[0].plot(param_val2, neg_llh2, label='comb', color='b', marker='.')
    # ax[0].set_ylabel('Total llh')
    # ax[0].legend()

    # ax[1].plot(param_val0, choice_llh0, label='ves', color='k', marker='.')
    # ax[1].plot(param_val1, choice_llh1, label='vis', color='r', marker='.')
    # ax[1].plot(param_val2, choice_llh2, label='comb', color='b', marker='.')
    # ax[1].set_ylabel('Choice llh')

    # ax[2].plot(param_val0, rt_llh0, label='ves', color='k', marker='.')
    # ax[2].plot(param_val1, rt_llh1, label='vis', color='r', marker='.')
    # ax[2].plot(param_val2, rt_llh2, label='comb', color='b', marker='.')
    # ax[2].set_ylabel('RT llh')






#  TESTING
#
# from ddm_moi.ddm_2d import (
#     set_params_dict_from_array, set_params_list, get_params_array_from_dict,
#     )


# init_params = {
#     'kmult': [0.1, 0.1, 0.3],
#     'bound': [0.6, 0.7, 0.7],
#     'ndt': [0.2, 0.3, 0.2],
#     'sigma_ndt': 0.05,
#     'theta': [1, 0.7, 0.7],
#     'alpha': 0.03,
#     }

# new_params = {
#     'kmult': [0.2, 0.2, 0.5],
#     'bound': [0.6, 0.7, 0.7],
#     'ndt': [0.2, 0.3, 0.2],
#     'sigma_ndt': 0.05,
#     'theta': [1, 0.5, 0.7],
#     'alpha': 0.05,
#     }

# param_keys = ['kmult', 'bound', 'ndt', 'sigma_ndt', 'theta', 'alpha']
# fixed = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
# init_params_array = get_params_array_from_dict(init_params, param_keys=param_keys)
# new_params_array = get_params_array_from_dict(new_params, param_keys=param_keys)

# params_array = set_params_list(new_params_array, init_params_array, fixed)

# # convert back to dict for passing to loss function
# params_dict = set_params_dict_from_array(params_array, init_params)


