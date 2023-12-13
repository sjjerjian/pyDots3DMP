# %% imports

import numpy as np

import pickle
import json

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

def test_simulation(nreps=200, full_save=False):
    return generate_sim_data(nreps=nreps, full_save=full_save)


def main_run_fit():

    # %% SIMULATED DATA

    with open(f"../data/ddm_2d_samp_100reps.pkl", "rb") as file:
        data, sim_params, log_odds_maps = pickle.load(file)
    data_delta0 = data.loc[(data['delta']==0), :] # bound hit and delta == 0

    # %% REAL DATA

    # filename = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230606.csv"
    # data = data_cleanup(filename)
    # data = format_onetargconf(data, remove_one_targ=True)
    # data_delta0 = data.loc[data['delta']==0, :]  # fit zero conflict data only

    # %% ----------------------------------------------------------------
    # Run first optimization for choice and RT only

    run_fit = True
    run_wager_fit = False

    accum = {'tvec': np.arange(0, 2, 0.01),  # 0.005
             'grid_vec': np.arange(-3, 0, 0.025),  # 0.025
             }

    # %% run the first fitting

    if run_fit:

        # initial parameter 'guesses'
        init_params = {
            'kmult': [0.15, 0.15, 0.25],
            'bound': [0.5, 0.5, 0.5],
            'ndt': [0.15, 0.15, 0.15],
            'sigma_ndt': 0.05,
        }

        lb = np.array([0.05, 0.05, 0.05, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.0])
        ub = np.array([0.3, 0.3, 0.5, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.1])

        plb = np.array([0.06, 0.06, 0.06, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.03])
        pub = np.array([0.2, 0.2, 0.4, 0.8, 0.8, 0.8, 0.3, 0.35, 0.3, 0.08])

        fixed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


        bads_result = run_bads(init_params, data_delta0, accum,
                               fixed=fixed, bounds=(lb, ub, plb, pub),
                               outputs=['choice', 'RT'])

        result_to_json(bads_result, "../data/lucio_bads_result_noPDW.json")

        fitted_params = ddm_2d.set_params_dict_from_array(bads_result.x, init_params)
        fitted_params['sigma_ndt'] = init_params['sigma_ndt']

        # for plotting user-set parameters, without fitting
        # fitted_params = {
        #     'kmult': [0.1, 0.1, 0.3],
        #     'bound': [0.6, 0.7, 0.7],
        #     'ndt': [0.2, 0.3, 0.2],
        #     'sigma_ndt': 0.05,
        #     # 'sigma_dv': 1,
        #     # 'theta': [1.1, 0.8, 0.9],  # in units of log odds
        #     # 'alpha': 0.03,
        # }

        # %% generate model predictions, based on fitted parameters

        ret_wager = False
        model_data, model_llh = model_predictions(fitted_params, data_delta0,
                                                  accum, num_hdgs=11,
                                                  return_wager=ret_wager)

        model_data_delta0 = model_data.loc[model_data['delta'] == 0, :].pipe(replicate_ves)

        plot_results(data_delta0, model_data_delta0, hue='modality',
                     return_wager=ret_wager)

    if run_wager_fit:
        # now rerun for theta and alpha (with tighter bounds on other params)
        # init_params2 = ddm_2d.set_params_dict_from_array(bads_result.x, init_params)
        # init_params2['sigma_ndt'] = init_params['sigma_ndt'] # reset to original
        # init_params2['theta'] = [0.8, 0.8, 0.8]
        # init_params2['alpha'] = 0.05

        init_params2 = {
            'kmult': [0.1, 0.1, 0.3],
            'bound': [0.6, 0.7, 0.7],
            'ndt': [0.2, 0.3, 0.2],
            'sigma_ndt': 0.05,
            'theta': [1, 0.7, 0.7],
            'alpha': 0.03,
            }


        lb = np.array([0.05, 0.05, 0.05, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0, 0.5, 0.5, 0.5, 0.0])
        ub = np.array([0.3, 0.3, 0.5, 0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.1, 1.5, 1.5, 1.5, 0.1])

        plb = np.array([0.06, 0.06, 0.06, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.02, 0.6, 0.6, 0.6, 0.03])
        pub = np.array([0.2, 0.2, 0.4, 0.8, 0.8, 0.8, 0.3, 0.35, 0.3, 0.05, 1.3, 1.3, 1.3, 0.08])

        # keep sigma_ndt and alpha fixed
        fixed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])

        bads_result2 = run_bads(init_params2, data_delta0, accum,
                                fixed=fixed, bounds=(lb, ub, plb, pub),
                                outputs=['choice', 'PDW', 'RT'])

        result_to_json(bads_result2, "../data/lucio_bads_result_wPDW.json")

        fitted_params2 = ddm_2d.set_params_dict_from_array(bads_result2.x, init_params2)
        fitted_params2['sigma_ndt'] = init_params['sigma_ndt']

        # %% generate model predictions, based on fitted parameters

        model_data, model_llh = model_predictions(fitted_params2, data_delta0, accum, num_hdgs=11, return_wager=True)
        model_data_delta0 = model_data.loc[model_data['delta']==0, :].pipe(replicate_ves)

        plot_results(data_delta0, model_data_delta0, hue='modality', return_wager=True)

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

    # TODO eventually initialize multiple times from different starting points, loop over function, add decorator?

    # lambda function to handle custom arguments
    target = lambda params: ddm_2d.objective(params, init_params, fixed,
                                            data=data, accum_kw=accum_kw,
                                            stim_scaling=True, method = 'prob',
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

def model_predictions(fitted_params, data, accum, num_hdgs=33, return_wager=True):

    # to plot a smooth curve of model prediction over headings
    max_hdg = 12
    # hdgs = np.array(np.linspace(-max_hdg, max_hdg, num_hdgs))
    hdgs = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]

    rt_method = 'peak'  # return the expected value of RT, given the model's prediction of the distribution

    conds = {'mods': [1, 2, 3],
             'cohs': [0.2, 0.7],
             'deltas': [0],
             'hdgs': hdgs}

    # generate model predictions using final fitted parameters on finely spaced headings
    # use the mean of the model RT distribution as predicted RT
    conds_for_pred = dots3DMP_create_conditions(conds, ['modality', 'coherence', 'delta', 'heading'])
    model_data, _, _ = ddm_2d.generate_data(params=fitted_params, data=conds_for_pred, accum_kw=accum,
                                        method='prob', rt_method='mean',
                                        stim_scaling=True, return_wager=return_wager)

    # and get the final error on the model
    fitted_params_array = ddm_2d.get_params_array_from_dict(fitted_params)
    fixed = np.ones_like(fitted_params_array)

    neg_llh = 0
    neg_llh = ddm_2d.objective(params=fitted_params_array,
                                init_params=fitted_params, fixed=fixed,
                                outputs=['choice', 'RT'], data=data,
                                accum_kw=accum, method='prob',
                                stim_scaling=True)

    return model_data, neg_llh


# %% ----------------------------------------------------------------
# Plot zero conflict data and model predictions (all modalities)

def plot_results(emp_data, model_data, hue, return_wager=True):

    # actual means from the data
    df_means = behavior_means(emp_data, by_conds=[hue, 'coherence', 'heading'], long_format=True).pipe(replicate_ves)
    df_means = df_means[df_means['variable'] != 'correct']

    if not return_wager:
        df_means = df_means[df_means['variable'] != 'PDW']

    palette = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    # palette = sns.color_palette("Set2", 3)

    g = plot_behavior_hdg(df_means, model_data, row='variable', col='coherence',
                          hue=hue, palette=palette)
    # TODO add legend

    return g

# ----------------------------------------------------------------
# helper function to save bads_results to a json file
def result_to_json(result, json_filename):

    del result['fun']  # can't be json-ified
    result['x'] = result['x'].tolist()
    result['x0'] = result['x0'].tolist()

    with open(json_filename, 'w') as json_file:
        json.dump(result, json_file, indent=4)

# %% ----------------------------------------------------------------
# ===== generate (and save) simulated data =====


def generate_sim_data(nreps=200, full_save=False):

    # TODO check if mvn rvs is random each time...or allow seed setting

    drop_no_hit = False

    sim_params = {
        'kmult': [0.1, 0.1, 0.3],
        'bound': [0.6, 0.7, 0.7],
        'ndt': [0.2, 0.3, 0.2],
        'sigma_ndt': 0.05,
        'sigma_dv': 1,
        'theta': [1.1, 0.8, 0.9],  # in units of log odds
        'alpha': 0.03,
    }

    conds = {'mods': [1, 2, 3],
             'cohs': [0.2, 0.7],
             'deltas': [-3, 0, 3],
             'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]}

    trial_table = dots3DMP_create_trial_list(**conds, nreps=nreps, shuff=False)

    accum = {'tvec': np.arange(0, 2, 0.01),  # 0.005
             'grid_vec': np.arange(-3, 0, 0.025), # 0.025
             }

    sim_method = 'samp'

    # return wager and use stim scaling by default
    save_dv = False
    return_wager = True
    sim_data, log_odds_maps, full_dvs = ddm_2d.generate_data(sim_params, data=trial_table, accum_kw=accum,
                                                             method=sim_method,
                                                             stim_scaling=True,
                                                             return_wager=return_wager,
                                                             save_dv=save_dv)

    if drop_no_hit:
        keep_trials = sim_data['bound_hit']
        sim_data = sim_data.loc[keep_trials, :]
        full_dvs = full_dvs[:, :, keep_trials]

    df_means = behavior_means(sim_data,
                              by_conds=['modality', 'coherence', 'heading', 'delta'],
                              drop_na=False,
                              long_format=True).pipe(replicate_ves)

    if not return_wager:
        df_means = df_means.loc[df_means['variable'] != 'PDW', :]

    df_means_delta0 = df_means.loc[df_means['delta']==0, :]
    df_means_deltas = df_means.loc[df_means['modality']==3, :]

    palette = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    g = plot_behavior_hdg(df_means_delta0,
                          row='variable', col='coherence', hue='modality',
                          palette=palette)

    g_d = plot_behavior_hdg(df_means_deltas,
                          row='variable', col='coherence', hue='delta',
                          palette=palette)

    if full_save:
        with open(f"../data/ddm_2d_{sim_method}_{nreps}reps.pkl", "wb") as file:
            pickle.dump((sim_data, sim_params, log_odds_maps), file, protocol=-1)
    else:
        sim_data.to_csv(f"../data/ddm_2d_{sim_method}_{nreps}reps.csv")

    return sim_data, log_odds_maps, full_dvs, sim_params


if __name__ == "__main__":
    # sim_data, log_odds_maps, full_dvs, sim_params = test_simulation(nreps=100, full_save=True)
    main_run_fit()





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


