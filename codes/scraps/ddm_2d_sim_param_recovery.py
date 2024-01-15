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

from datetime import datetime

# %% 

# USER SETTINGS
nreps = 100

conds = {'mods': [1, 2, 3],
         'cohs': [0.3, 0.7],
         'deltas': [-3, 0, 3],
         'hdgs': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]}

trial_table = dots3DMP_create_trial_list(**conds, nreps=nreps, shuff=False)

accum_kw = {'tvec': np.arange(0, 2, 0.01),  # 0.005
            'grid_vec': np.arange(-3, 0, 0.025), # 0.025
            }

stim_scaling = ddm_2d.get_stim_urgs(tvec=accum['tvec'])

init_params = {
    'kmult': [0.15, 0.4],
    'bound': [0.5, 0.5, 0.5],
    'ndt': 0.25,
    'sigma_ndt': 0.05,
    'theta': [1, 1, 1],  # in units of log odds
    'alpha': 0.05,
}


# %% ----------------------------------------------------------------

def generate_fake_data():
    
    sim_params = {
        'kmult': [0.35, 0.35, 1.4],
        'bound': [0.6, 0.75, 1.5],
        'ndt': 0.25,
        'sigma_ndt': 0.05,
        'sigma_dv': 1,
        'theta': [1.2, 0.9, 1.1],  # in units of log odds
        'alpha': 0.05,
    }
        
    # %% run sim and save results

    # TODO: save full_dvs option
    sim_data, wager_maps, full_dvs = ddm_2d.generate_data(
        sim_params, data=trial_table, accum_kw=accu_kw,
        pred_method='sample', stim_scaling=True, return_wager=return_wager)

    cur_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'dots3DMP_ddm_2d_fake_data_{nreps}reps_{cur_dt}'

    # save all outputs into pkl file, and sim_data DataFrame into csv too
    with open(f"../data/{filename}.pkl", "wb") as file:
        pickle.dump((sim_data, sim_params, wager_maps), file, protocol=-1)
    sim_data.to_csv(f"../data/{filename}.csv")

    # %% plot summary

    df_means = behavior_means(
        sim_data, by_conds=['modality', 'coherence', 'heading', 'delta'],
        drop_na=False, long_format=True).pipe(replicate_ves)

    # zero conflict + unimodal conditions
    df_means_delta0 = df_means.loc[df_means['delta'] == 0, :]
    palette = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]
    plot_behavior_hdg(
        df_means_delta0, row='variable', col='coherence', hue='modality', palette=palette)

    # cue conflict
    df_means_deltas = df_means.loc[df_means['modality'] == 3, :]
    palette = [(0, 0, 1), (0, 1, 1), (0, 1, 0)]
    plot_behavior_hdg(
        df_means_deltas, row='variable', col='coherence', hue='delta', palette=palette)


def run_model_fit(data_for_fit: pd.DataFrame, run_fit=True):
    # %% load simulated data and set initial parameters

    sim_data = pd.read_csv(f"../data/{filename}.csv")

    # TODO loop over init_params for multiple starting points
    # TODO may be better to provide fixed params as separate input and stitch together inside

    init_params_array = ddm_2d.get_params_array_from_dict(init_params)

    # lambda function to handle extra arguments
    target = lambda params: ddm_2d.objective(params, init_params, fixed, data=data_for_fit, 
                                            accum_kw=accum_kw, stim_scaling=stim_scaling, 
                                            pred_method='proba', outputs=outputs)

    bads = BADS(target, init_params_array, lb, ub, plb, pub)
    bads_result = bads.optimize()
            
    # and get fitted params back as dictionary from bads result object
    init_params_array, fitted_params_array = bads_result.x0, bads_result.x
    fitted_params_array[fixed == 1] = init_params_array[fixed == 1]
    fitted_params = ddm_2d.set_params_dict_from_array(fitted_params_array, init_params)
    
    return fitted_params

    
    
    # %% (re-)run with final parameters to get predictions and wager_odds_maps (if needed)
    neg_llh, model_llh, model_data, wager_maps = ddm_2d.get_llhs(
        params=fitted_params, data=data_for_fit, accum_kw=accum_kw,
        pred_method='proba', rt_pred='lik',
        stim_scaling=stim_scaling,
    )
    print(f'Final log likelihood: {neg_llh:.2f}')

    # generate model predictions using final fitted parameters 
    # (this can then use a different set of conditions (e.g. finely spaced headings, deltas) to what we fitted
    # but we have to use the pre-calculated log odds maps!

    model_data, _, _ = ddm_2d.generate_data(
        params=fitted_params, data=pred_conds, accum_kw=accum_kw, 
        pred_method='proba', rt_pred='mean',
        stim_scaling=stim_scaling,
        wager_odds_maps=wager_maps,
        )
    
    return model_data




if __name__ == '__main__':
    
    generate_fake_data()
    
    data = pd.read_csv()







