"""Main functions for generating/fitting 2-D DDM to dots3DMP task."""

import numpy as np
import pandas as pd

from scipy.signal import convolve
from scipy.stats import norm, truncnorm, skewnorm
from collections import OrderedDict, namedtuple

from typing import Optional, Sequence
from functools import wraps
from codetiming import Timer

from ddm_moi.Accumulator import AccumulatorModelMOI

def optim_decorator(loss_func):
    """
    Optimization tools like pybads require a loss function as input which returns only a single value.

    Since our overall loss function and internals handle a bunch of inputs and outputs, this decorator
    is designed to 'wrap' the loss function so that we can easily optimisze over the single loss value,
    while still handling the other parameters and outputs

    ***You do not need to use this function directly, instead just call the loss function and it will be
    passed through this decorator, returning the current set of parameters and their associated loss.***
    """

    # TODO figure out if where we are setting the fixed parameters is the "right" place
    # i.e. is the optimization process still trying to find a value for them? I think it might be...
    # could try to pass in a fixed params dict which gets merged, but then need to know
    # what the param_keys are anyway...might be worth revisiting this in tandem with the ordering issue below

    @wraps(loss_func)
    def wrapper(params: np.ndarray, init_params: OrderedDict, fixed: np.ndarray = None, *args, **kwargs):

        if fixed is None:
            fixed = np.zeros_like(params)

        # Actually params MUST be set in this order currently, since we don't
        # reindex "fixed" to match if they are not...
        param_keys = ['kmult', 'bound', 'ndt', 'sigma_ndt']
        if 'PDW' in kwargs['outputs']:
            param_keys = ['kmult', 'bound', 'ndt', 'sigma_ndt', 'theta', 'alpha']

        # convert initial params to array, replace params where fixed is true
        init_params_array = get_params_array_from_dict(init_params, param_keys=param_keys)
        params_array = set_params_list(params, init_params_array, fixed)

        # convert back to dict for passing to loss function
        params_dict = set_params_dict_from_array(params_array, init_params)

        # print out the current param vals
        for key, val in params_dict.items():
            print(f"{key}: {val}\t")

        loss_val, llhs, model_data, wager_odds_maps = loss_func(params_dict, *args, **kwargs)

        if loss_val == np.inf:
            print(params_dict)
            raise ValueError("loss function evaluated to infinite")
        else:
            return loss_val

    return wrapper


@optim_decorator
def objective(params: dict, data: pd.DataFrame, outputs: Optional[Sequence] = None,
              llh_scaling: Optional[Sequence] = None, **gen_data_kwargs):
    """
    Calculate individual log-likelihood for each behavioral outcome, given parameters and data.

    Choice and PDW log-likelihoods are calculated according to binomial probability
    RT log-likelihoods are calculated based on the full RT distribution
    Total log likelihood is a (weighted) sum of individual log likelihoods.
    Optional arguments outputs and llh_scaling determine which outcomes contribute and their
    weighting. Default is all three weighted equally.

    Args:
        params (dict): a dictionary containing the current set of parameters
        data (pd.DataFrame): actual data for calculating the log likelihood of the parameters on
        outputs (Sequence, optional): which behavior outputs do ca. Defaults to [choice, PDW, RT]
        llh_scaling (Sequence, optional): relative weighting for each output.
                                        Defaults to all being equal i.e. [1, 1, 1].

    Returns
    -------
        neg_llh: the total negative log-likelihood
        model_llh: individual negative log-likelihoods for each output
        model_data: the model predictions for each output given the trial conditions in data
        wager_odds_maps: the log-odds maps used for PDW predictions for the given set of parameters
                        these can be optionally passed back in to another call to generate_data
                        to use a previously calculated log-odds map
    """
    if outputs is None:
        outputs = ['choice', 'PDW', 'RT']

    if llh_scaling is None:
        llh_scaling = np.ones(len(outputs))

    assert len(outputs) == len(llh_scaling), 'outputs and their llh weights should match in length'

    # only calculate pdfs if fitting confidence variable
    return_wager = 'PDW' in outputs

    # get model predictions (probabilistic) given parameters
    model_data, wager_maps, _ = generate_data(params=params, data=data,
                                                 return_wager=return_wager,
                                                 **gen_data_kwargs)

    # calculate log likelihoods of parameters, given observed data
    model_llh = dict()

    # choice and PDW likelihoods according to bernoulli probability
    model_llh['choice'] = \
        np.sum(np.log(model_data.loc[data['choice'] == 1, 'choice'])) + \
        np.sum(np.log((1 - model_data.loc[data['choice'] == 0, 'choice'])))

    if return_wager:
        model_llh['PDW'] = \
            np.sum(np.log(model_data.loc[data['PDW'] == 1, 'PDW'])) + \
            np.sum(np.log(1 - model_data.loc[data['PDW'] == 0, 'PDW']))

    # RT likelihood, straight from dataframe
    model_llh['RT'] = np.log(model_data['RT']).sum()

    # sum the individual log likelihoods included in outputs,
    # (after scaling them according to llh_scaling)
    neg_llh = -sum([model_llh[v]*w for v, w in zip(outputs, llh_scaling) if v in model_llh])

    print(f"Total loss:{neg_llh:.2f}")
    print({key: round(model_llh[key], 2) for key in model_llh})
    print('\n\n')
    
    model_results = namedtuple('ModelResults', ['neg_llh', 'llhs', 'model_data', 'wager_maps'])
    return model_results(neg_llh, llhs, model_data, wager_maps)

# get the wrapped function out for when we want to call it without running the optimization!
get_llhs = objective.__wrapped__


@Timer(name="ddm_run_timer")
def generate_data(params: dict, data: pd.DataFrame, accum_kw: dict,
                  pred_method: Optional[str] = 'proba', rt_method: Optional[str] = 'lik',
                  save_dv: [bool] = False, stim_scaling: Optional[bool, tuple] = True, 
                  cue_weights: Optional[tuple, str] = 'optimal', seed: Optional[int] = None, 
                  return_wager: [bool] = True, wager_odds_maps=None, wager_thres: str = 'log_odds') -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generates model outputs for behavioral variables given parameters and trial conditions.
    For producing behavioral outputs (choice, PDW, RT), several options are available.
    
    pred_method == 'proba', 'sim', or 'sample'
    
    rt_pred == 'lik', 'mean', 'max' - returns the likelihood of observed RTs, or the mean/max of the RT distribution
    """
 
    # ---- input handling ----
    assert pred_method == 'proba' or pred_method == 'sim_dv' or pred_method == 'sample', "pred_method must be one of 'proba', 'sim_dv', or 'sample'"
    assert rt_pred == 'mean' or rt_pred == 'max' or rt_pred == 'lik', "return_type must be one of 'lik', 'mean', 'max'"
    assert cue_weights == 'optimal' or cue_weights == 'random' or (isinstance(cue_weights, tuple) and len(cue_weights)==2), \
        "cue_weights must be one of 'optimal', 'random', or a tuple of length 2"
    assert wager_thres == 'log_odds' or wager_thres == 'time' or wager_thres == 'evidence', \
        "wager_thres must be one of 'log_odds', 'time' or 'evidence"
        
    if seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # initialize model_data output dataframe
    mods = np.unique(data['modality'])
    cohs = np.unique(data['coherence'])
    hdgs = np.unique(data['heading'])
    deltas = np.unique(data['delta'])

    model_data = data.loc[:, ['heading', 'coherence', 'modality', 'delta']]
    model_data[['choice', 'PDW', 'RT']] = np.nan

    # initialize accumulator object
    accumulator = AccumulatorModelMOI(**accum_kw)
    accumulator.bound = params['bound']

    # we'll need this later
    orig_tvec = accumulator.tvec
    orig_dt = np.diff(orig_tvec[:2]).item()

    full_dvs = None
    if pred_method == 'sim_dv' and save_dv:
        full_dvs = np.full((len(accumulator.tvec), 2, len(model_data)), np.nan)
    else:
        save_dv = False

    # set up sensitivity and other parameters
    if wager_odds_maps is not None:
        assert isinstance(wager_odds_maps, list) and len(wager_odds_maps) == len(mods), \
            "wager_odds_maps must be a list equal to the number of unique modalities"

    # replicate ndt for each modality if a single value is provided
    if isinstance(params['ndt'], (int, float)):
        params['ndt'] = [params['ndt']] * len(mods)

    # for defining an NDT distribution and then convolving with RT, we need an SD > 0
    if ('sigma_ndt' not in params) or (params['sigma_ndt'] == 0):
        params['sigma_ndt'] = 1e-100

    if not stim_scaling:
        # no stim scaling by sensitivity, b_ves and b_vis are just ones
        b_ves, b_vis = np.ones_like(accumulator.tvec), np.ones_like(accumulator.tvec)
        kmult_scaling = 10

    else:
        # use provided stim profiles, or get the default set ones in get_stim_urgs
        if isinstance(stim_scaling, tuple):
            b_vis, b_ves = stim_scaling
        elif stim_scaling is True:
            b_vis, b_ves = get_stim_urgs(tvec=accumulator.tvec)

        # new elapsed time is squared accumulated time course of sensitivity
        cumul_bves = np.cumsum((b_ves**2)/(b_ves**2).sum())
        cumul_bvis = np.cumsum((b_vis**2)/(b_vis**2).sum())
        kmult_scaling = 1

    # to be nx1 arrays, for later convenience
    b_ves, b_vis = b_ves.reshape(-1, 1), b_vis.reshape(-1, 1)

    # scale back up, because we downweighted the input to keep all parameters at a similar order of magnitude
    kmult = [k * kmult_scaling for k in params['kmult']]

    # set kves and kvis, based on length of kmult
    if isinstance(kmult, (int, float)) or len(kmult) == 1:
        kvis = kmult * cohs.T
        kves = np.mean(kvis)  # ves lies between vis cohs
    else:
        kves = kmult[0]
        if len(kmult) == 2:
            kvis = kmult[1] * cohs.T
        else:
            kvis = kmult[1:]  # 3 independent kmults

    # ====== generate pdfs and log_odds for confidence ======
    # looping over modalities
    # but marginalize over cohs, and ignore delta (assume confidence mapping is stable across these)
    # sensitivity is determined not just be k (internal sensivitity) but also b (external sensivitity)
    # i.e. time-dependent stimulus reliability

    if return_wager:
        if isinstance(params['theta'], (int, float)):
            params['theta'] = [params['theta']] * len(mods)

        if wager_odds_maps is None:

            wager_odds_maps = []
            sin_uhdgs = np.sin(np.deg2rad(hdgs[hdgs >= 0]))

            for m, mod in enumerate(mods):

                if isinstance(params['bound'], (list, np.ndarray)) and len(params['bound']) == 3:
                    accumulator.bound = params['bound'][m]

                if mod == 1:
                    # Drugo 2014 supplementary materials:
                    # if we apply "stimulus-scaling", the passage of time becomes the cumulative
                    # sensitivity of the stimulus
                    # b(t) is also within the momentary evidence term, hence the squaring,
                    # and this deals with any negatives (e.g. in acceleration)

                    if stim_scaling:
                        accumulator.tvec = cumul_bves * orig_tvec[-1]
                        abs_drifts = np.cumsum(b_ves**2 * kves * sin_uhdgs, axis=0)
                    else:
                        abs_drifts = kves * sin_uhdgs

                elif mod == 2:
                    if stim_scaling:
                        accumulator.tvec = cumul_bvis * orig_tvec[-1]
                        abs_drifts = np.cumsum(b_vis**2 * np.mean(kvis) * sin_uhdgs, axis=0)
                    else:
                        abs_drifts = np.mean(kvis) * sin_uhdgs

                elif mod == 3:
                    
                    if cue_weights == 'optimal':
                        kves2, kvis2 = kves**2, np.mean(kvis)**2
                        kcomb2 = kves2 + kvis2
                        w_ves = np.sqrt(kves2 / kcomb2)
                        w_vis = np.sqrt(kvis2 / kcomb2)
                    elif cue_weights == 'random':
                        w_ves = rng.rand()
                        w_vis = rng.rand()
                    elif isinstance(cue_weights, tuple):
                        w_ves, wvis = cue_weights
                        
                    # Eq 14
                    if stim_scaling:
                        t_comb = w_ves**2 * cumul_bves + w_vis**2 * cumul_bvis
                        accumulator.tvec = t_comb * orig_tvec[-1]

                        drift_ves = np.cumsum(b_ves**2 * kves * sin_uhdgs, axis=0)
                        drift_vis = np.cumsum(b_vis**2 * np.mean(kvis) * sin_uhdgs, axis=0)
                        abs_drifts = w_ves * drift_ves + w_vis * drift_vis
                    else:
                        abs_drifts = np.sqrt(kcomb2).reshape(-1, 1) * sin_uhdgs

                # divide by the new passage of time if we applied the scaling, because we will
                # re-multiply by it in the Accumulator Class logic,
                # and this will yield the *position* of the particle
                if stim_scaling:
                    abs_drifts /= accumulator.tvec.reshape(-1, 1)

                if abs_drifts.ndim == 2:
                    drifts_list = [abs_drifts[:, i:i+1] for i in range(abs_drifts.shape[1])]
                else:
                    drifts_list = abs_drifts.tolist()

                # run the method of images dtb to extract pdfs, cdfs, and LPO
                accumulator.set_drifts(drifts_list, hdgs[hdgs >= 0])
                accumulator.dist(return_pdf=True).log_posterior_odds()
                wager_odds_maps.append(accumulator.log_odds)

                # allow for different configurations of confidence mapping
                if wager_thres == 'log_odds':
                    wager_odds_above_threshold = [p >= theta for p, theta in zip(wager_odds_maps, params['theta'])]
                elif wager_thres == 'time':
                    raise NotImplementedError
                elif wager_thres == 'evidence':
                    raise NotImplementedError
                # TODO just replace wager_odds_maps with a repmat of the grid or time vectors
                # but probably need to use < to make sure the interpretation/sign of theta is consistent
                    # wager_odds_above_threshold = [p >= theta for p, theta in zip(wager_odds_maps, params['theta'])]


    # ===== now to generate model results, loop over all conditions =====
    # here we used the signed drift rates, to allow for asymmetries e.g. because of cue conflict
    # cue sensivities are the same, except we use individual headings AND coherences now, rather than marginalizing across them

    for m, mod in enumerate(mods):

        if isinstance(params['bound'], (list, np.ndarray)) and len(params['bound']) == 3:
            accumulator.bound = params['bound'][m]

        ndt_min = params['ndt'][m] / 2
        ndt_max = params['ndt'][m] + ndt_min

        for c, coh in enumerate(cohs):
            for d, delta in enumerate(deltas):

                # non-valid conditions
                if (delta != 0 and mod < 3) or (c > 0 and mod == 1):
                    continue

                # print(mod, coh, delta)

                if mod == 1:
                    if stim_scaling:
                        accumulator.tvec = cumul_bves * orig_tvec[-1]
                        drifts = np.cumsum(b_ves**2 * kves * np.sin(np.deg2rad(hdgs)), axis=0)
                    else:
                        drifts = kves * np.sin(np.deg2rad(hdgs))

                elif mod == 2:
                    if stim_scaling:
                        accumulator.tvec = cumul_bvis * orig_tvec[-1]
                        drifts = np.cumsum(b_vis**2 * kvis[c] * np.sin(np.deg2rad(hdgs)), axis=0)
                    else:
                        drifts = kvis * np.sin(np.deg2rad(hdgs))

                elif mod == 3:

                    if cue_weights == 'optimal':
                        kves2, kvis2 = kves**2, kvis[c]**2
                        kcomb2 = kves2 + kvis2
                        w_ves = np.sqrt(kves2 / kcomb2)
                        w_vis = np.sqrt(kvis2 / kcomb2)
                    elif cue_weights == 'random' and not return_wager:
                        # only generate new ones if we didn't generate them already for the log odds maps!
                        w_ves = rng.rand()
                        w_vis = rng.rand()
                        # TODO add logger to return the cue_weights so we know what they were!
                    elif isinstance(cue_weights, tuple):
                        w_ves, wvis = cue_weights

                    # Eq 14
                    if stim_scaling:
                        t_comb = w_ves**2 * cumul_bves + w_vis**2 * cumul_bvis
                        accumulator.tvec = t_comb * orig_tvec[-1]

                        # +ve delta means ves to the left, vis to the right
                        # read the eq as cumsum each modality separately first, then do the weighted sum
                        drift_ves = np.cumsum(b_ves**2 * kves * np.sin(np.deg2rad(hdgs - delta / 2)), axis=0)
                        drift_vis = np.cumsum(b_vis**2 * kvis[c] * np.sin(np.deg2rad(hdgs + delta / 2)), axis=0)

                    else:
                        drift_ves = kves * np.sin(np.deg2rad(hdgs - delta / 2))
                        drift_vis = kvis[c] * np.sin(np.deg2rad(hdgs + delta / 2))

                    # Eq 10/17
                    drifts = w_ves * drift_ves + w_vis * drift_vis

                if pred_method == 'sim_dv':
                    drifts = np.gradient(drifts, axis=0)
                    drifts *= orig_dt

                    sigma_dv = params['sigma_dv'] * np.sqrt(orig_dt)
                    sigma_dv = np.array([sigma_dv, sigma_dv])

                elif stim_scaling:
                    # (since we'll remultiply by tvec in the cdf/pdf code)
                    # but if simulating dv we want to keep it as is
                    drifts /= accumulator.tvec.reshape(-1, 1)

                # calculate cdf and pdfs using signed drift rates now
                if drifts.ndim == 2:
                    drifts_list = [drifts[:, i:i+1] for i in range(drifts.shape[1])]
                else:
                    drifts_list = drifts.tolist()
                accumulator.set_drifts(drifts_list, hdgs)

                # run the MOI to get the CDF and PDF results
                if pred_method != 'sim_dv':
                    accumulator.dist(return_pdf=return_wager)

                for h, hdg in enumerate(hdgs):
                    trial_index = (data['modality'] == mod) & (data['coherence'] == coh) & \
                                  (data['heading'] == hdg) & (data['delta'] == delta)

                    if trial_index.sum() == 0:
                        continue

                    # simulate decision variables
                    if pred_method == 'sim_dv':
                        these_trials = np.where(trial_index)[0]

                        non_dec_time = truncnorm.rvs(ndt_min, ndt_max, loc=params['ndt'][m], scale=params['sigma_ndt'],
                                                     size=trial_index.sum())

                        for tr in range(trial_index.sum()):
                            dv = accumulator.dv(drift=accumulator.drift_rates[h], sigma=sigma_dv)

                            bound_crossed = (dv >= accumulator.bound).any(axis=0)
                            time_crossed = np.argmax((dv >= accumulator.bound) == 1, axis=0)

                            model_data.loc[these_trials[tr], 'bound_hit'] = bound_crossed.any()

                            if bound_crossed.all():
                                # both accumulators hit the bound - choice and RT determined by whichever hit first
                                choice = np.argmin(time_crossed)
                                rt_ind = np.min(time_crossed)
                                final_v = dv[rt_ind, choice ^ 1]  # losing accumulator

                            elif ~bound_crossed.any():
                                # neither accumulator hits the bound
                                rt_ind = -1
                                choice = np.argmax(np.argmax(dv, axis=0))  # winner is whoever has max dv value
                                final_v = accumulator.bound[choice] - (dv[rt_ind, choice] - dv[rt_ind, choice ^ 1])
                                # wager odds map accounts for the distance between winner and loser, so this shifts up both accumulators
                                # as if the 'winner' did hit the bound, so we can do a consistent look-up on the wager odds map

                            else:
                                # only one hits the bound
                                choice = np.where(bound_crossed)[0].item()
                                rt_ind = time_crossed[choice]
                                final_v = dv[rt_ind, choice ^ 1]

                            if return_wager:

                                # look-up log odds threshold
                                grid_ind = np.argmin(np.abs(accumulator.grid_vec - final_v))
                                # log_odds = wager_odds_maps[m][rt_ind, grid_ind]
                                wager = int(wager_odds_above_threshold[m][rt_ind, grid_ind])
                                wager *= (np.random.random() > params['alpha'])  # incorporate base-rate of low bets
                                model_data.loc[these_trials[tr], 'PDW'] = wager

                            # flip choice result so that left choices = 0, right choices = 1 in the output
                            model_data.loc[these_trials[tr], 'choice'] = choice ^ 1

                            # RT = decision time + non-decision time - motion onset latency
                            model_data.loc[these_trials[tr], 'RT'] = \
                                orig_tvec[rt_ind] + non_dec_time[tr] - 0.3

                            if save_dv:
                                full_dvs[:, :, these_trials[tr]] = dv

                    else:
                        # otherwise, get model generated predictions
                        # then either return those directly as probabilities
                        # or sample from them to generate 'trials'

                        p_right = accumulator.p_corr[h].item()
                        p_right = np.clip(p_right, 1e-100, 1-1e-100)
                        p_choice = np.array([p_right, 1 - p_right])
                        
                        # ====== CHOICE ======
                        # if pred_method is probability, we just store the predicted p_right by the model
                        # if pred_method is sample, then generate ntrial Bernoulli samples according to the p_choice right/left distribution
                        if pred_method == 'proba':
                            model_data.loc[trial_index, 'choice'] = p_right
                        elif pred_method == 'sample':
                            model_data.loc[trial_index, 'choice'] = \
                                rng.choice([1, 0], trial_index.sum(), replace=True, p=p_choice)
                                # rng.binomial(n=1, p=p_right, size=trial_index.sum()) # should be the same thing...

                        # ====== WAGER ======
                        if return_wager:
                            # select pdf for losing race, given correct or incorrect
                            pxt_up = np.squeeze(accumulator.up_lose_pdf[h, :, :])
                            pxt_lo = np.squeeze(accumulator.lo_lose_pdf[h, :, :])
                            total_p = np.sum(pxt_up + pxt_lo)
                            pxt_up /= total_p
                            pxt_lo /= total_p

                            p_choice_and_wager = np.array(
                                [[np.sum(pxt_up[wager_odds_above_threshold[m]]),     # pRight+High
                                  np.sum(pxt_up[~wager_odds_above_threshold[m]])],   # pRight+Low
                                 [np.sum(pxt_lo[wager_odds_above_threshold[m]]),     # pLeft+High
                                  np.sum(pxt_lo[~wager_odds_above_threshold[m]])]],  # pLeft+Low
                            )

                            # calculate p_wager using Bayes rule, then factor in base rate of low bets ("alpha")
                            p_choice_given_wager, p_wager = _margconds_from_intersection(p_choice_and_wager, p_choice)
                            p_wager += np.array([-params['alpha'], params['alpha']]) * p_wager[0]
                            p_wager = np.clip(p_wager, 1e-100, 1-1e-100)

                            if pred_method == 'proba':
                                model_data.loc[trial_index, 'PDW'] = p_wager[0]
                            elif pred_method == 'sample':
                                model_data.loc[trial_index, 'PDW'] = rng.choice(
                                    [1, 0], trial_index.sum(), replace=True, p=p_wager)
                                # rng.binomial(n=1, p=p_wager[0], size=trial_index.sum()) # should be the same thing...


                        # ====== RT ======

                        # first convolve model RT distribution with non-decision time
                        ndt_dist = norm.pdf(orig_tvec, loc=params['ndt'][m], scale=params['sigma_ndt'])
                        rt_dist = np.squeeze(accumulator.rt_dist[h, :])

                        rt_dist = np.clip(rt_dist, np.finfo(np.float64).eps, a_max=None)
                        rt_dist = convolve(rt_dist, ndt_dist / ndt_dist.sum())
                        rt_dist = rt_dist[:len(accumulator.tvec)]
                        rt_dist /= rt_dist.sum()  # renormalize

                        # given RT distribution,
                        #   sample from distribution (generating fake RTs) if return_proba,
                        #   otherwise store probability of observed RTs, or return mean/max of model-predicted RT distribution

                        # NOTE 12-2023 I think we need to sample from the orig_tvec here, not the changed tvec!
                        # TODO remove hard-coded 0.3 by calculating time of 1% of max acceleration (in experiment this is where RT is calculated from)

                        if pred_method == 'sample':
                            rt_output = rng.choice(orig_tvec, trial_index.sum(), replace=True, p=rt_dist) - 0.3
                        else:
                            if rt_method == 'likelihood':
                                # NOTE in this case we save the likelihood values, not an RT value!
                                actual_rts = data.loc[trial_index, 'RT'].values + 0.3
                                dist_inds = [np.argmin(np.abs(orig_tvec - rt)) for rt in actual_rts]
                                rt_dist /= rt_dist.max()  # rescale to 0-1, makes likelihood total closer in magnitude to choice and PDW
                                rt_output = rt_dist[dist_inds]
                            elif rt_method == 'mean':
                                rt_output = (orig_tvec * rt_dist).sum() - 0.3
                            elif rt_method == 'peak':
                                rt_output = orig_tvec[np.argmax(rt_dist)] - 0.3

                        model_data.loc[trial_index, 'RT'] = rt_output
                        
    # to avoid log(0) issues when doing log-likelihoods, replace zeros and ones
    if return_proba:
        model_data.loc[:, ['choice', 'PDW']] = model_data.loc[:, ['choice', 'PDW']].replace(to_replace=0, value=1e-10)
        model_data.loc[:, ['choice', 'PDW']] = model_data.loc[:, ['choice', 'PDW']].replace(to_replace=1, value=1 - 1e-10)

    return model_data, wager_odds_maps, full_dvs


# @Timer(name="accumulator_timer")
def dots3dmp_accumulator(params: dict, hdgs, mod, delta, accum_kw: dict,
                  stim_scaling=True, use_signed_drifts: bool = True) -> AccumulatorModelMOI:
    """
    pared down version of generate_data, to return the actual accumulator object for given set of conditions
    NOTE provide kmult param as a 2-element vector, for kves and kvis
    
    TODO consider whether this can be implemented within generate_data, or whether generate_data needs to be broken down.

    """
    # initialize accumulator
    accumulator = AccumulatorModelMOI(**accum_kw)
    accumulator.bound = params['bound']

    # we'll need this later
    orig_tvec = accumulator.tvec
    orig_dt = np.diff(orig_tvec[:2]).item()

    # set up sensitivity and other parameters

    if not stim_scaling:
        b_ves, b_vis = np.ones_like(accumulator.tvec), np.ones_like(accumulator.tvec)
        kmult_scaling = 10

    else:
        if isinstance(stim_scaling, tuple):
            b_vis, b_ves = stim_scaling
        elif stim_scaling is True:
            b_vis, b_ves = get_stim_urgs(tvec=accumulator.tvec)

        # new elapsed time is squared accumulated time course of sensitivity
        cumul_bves = np.cumsum((b_ves**2)/(b_ves**2).sum())
        cumul_bvis = np.cumsum((b_vis**2)/(b_vis**2).sum())

        kmult_scaling = 1

    # to be nx1 arrays
    b_ves, b_vis = b_ves.reshape(-1, 1), b_vis.reshape(-1, 1)

    # scale up, because we downweighted the input to parameters at a similar order of magnitude

    kmult = [k*kmult_scaling for k in params['kmult']]

    # set kves and kvis
    kves, kvis = kmult[0], kmult[1]

    # ====== generate pdfs and log_odds for confidence ======
    # ignore delta (assume confidence mapping is stable across these)
    # sensitivity is determined not just be k (internal sensivitity) but also b (external sensivitity)
    # i.e. time-dependent stimulus reliability

    if not use_signed_drifts:

        sin_uhdgs = np.sin(np.deg2rad(hdgs[hdgs >= 0]))

        if mod == 1:
            # Drugo 2014 supplementary materials:
            # if we apply "stimulus-scaling", the passage of time becomes the cumulative sensitivsity of the stimulus
            # b(t) is also within the momentary evidence term, hence the squaring,
            # and this deals with any negatives (e.g. in acceleration)

            if stim_scaling:
                accumulator.tvec = cumul_bves * orig_tvec[-1]
                abs_drifts = np.cumsum(b_ves**2 * kves * sin_uhdgs, axis=0)
            else:
                abs_drifts = kves * sin_uhdgs

        elif mod == 2:
            if stim_scaling:
                accumulator.tvec = cumul_bvis * orig_tvec[-1]
                abs_drifts = np.cumsum(b_vis**2 * kvis * sin_uhdgs, axis=0)
            else:
                abs_drifts = kvis * sin_uhdgs

        elif mod == 3:
            kves2, kvis2 = kves**2, kvis**2
            kcomb2 = kves2 + kvis2

            w_ves = np.sqrt(kves2 / kcomb2)
            w_vis = np.sqrt(kvis2 / kcomb2)

            # Eq 14
            if stim_scaling:
                t_comb = w_ves**2 * cumul_bves + w_vis**2 * cumul_bvis
                accumulator.tvec = t_comb * orig_tvec[-1]

                drift_ves = np.cumsum(b_ves**2 * kves * sin_uhdgs, axis=0)
                drift_vis = np.cumsum(b_vis**2 * kvis * sin_uhdgs, axis=0)
                abs_drifts = w_ves * drift_ves + w_vis * drift_vis
            else:
                abs_drifts = np.sqrt(kcomb2).reshape(-1, 1) * sin_uhdgs

        # divide by the new passage of time if we applied the scaling, because we will re-multiply by it in the Accumulator Class logic,
        # and this will yield the *position* of the particle
        if stim_scaling:
            abs_drifts /= accumulator.tvec.reshape(-1, 1)

        if abs_drifts.ndim == 2:
            drifts_list = [abs_drifts[:, i:i+1] for i in range(abs_drifts.shape[1])]
        else:
            drifts_list = abs_drifts.tolist()

        # run the method of images dtb to extract pdfs, cdfs, and LPO
        accumulator.set_drifts(drifts_list, hdgs[hdgs >= 0])
        accumulator.dist(return_pdf=True).log_posterior_odds()

    else:

        # ===== calculate accumulator outputs =====
        # here we used the signed drift rates, to allow for asymmetries e.g. because of cue conflict

        if mod == 1:
            if stim_scaling:
                accumulator.tvec = cumul_bves * orig_tvec[-1]
                drifts = np.cumsum(b_ves**2 * kves * np.sin(np.deg2rad(hdgs)), axis=0)
            else:
                drifts = kves * np.sin(np.deg2rad(hdgs))

        elif mod == 2:
            if stim_scaling:
                accumulator.tvec = cumul_bvis * orig_tvec[-1]
                drifts = np.cumsum(b_vis**2 * kvis * np.sin(np.deg2rad(hdgs)), axis=0)
            else:
                drifts = kvis * np.sin(np.deg2rad(hdgs))

        elif mod == 3:

            kves2, kvis2 = kves**2, kvis**2
            kcomb2 = kves2 + kvis2

            w_ves = np.sqrt(kves2 / kcomb2)
            w_vis = np.sqrt(kvis2 / kcomb2)

            # Eq 14
            if stim_scaling:
                t_comb = w_ves**2 * cumul_bves + w_vis**2 * cumul_bvis
                accumulator.tvec = t_comb * orig_tvec[-1]

                # +ve delta means ves to the left, vis to the right
                # read the eq as cumsum each modality separately first, then do the weighted sum
                drift_ves = np.cumsum(b_ves**2 * kves * np.sin(np.deg2rad(hdgs - delta / 2)), axis=0)
                drift_vis = np.cumsum(b_vis**2 * kvis * np.sin(np.deg2rad(hdgs + delta / 2)), axis=0)

            else:
                drift_ves = kves * np.sin(np.deg2rad(hdgs - delta / 2))
                drift_vis = kvis * np.sin(np.deg2rad(hdgs + delta / 2))

            # Eq 10/17
            drifts = w_ves * drift_ves + w_vis * drift_vis

        if stim_scaling:
            # since we'll remultiply by tvec in the cdf/pdf code
            drifts /= accumulator.tvec.reshape(-1, 1)

        # calculate cdf and pdfs using signed drift rates now
        if drifts.ndim == 2:
            drifts_list = [drifts[:, i:i+1] for i in range(drifts.shape[1])]
        else:
            drifts_list = drifts.tolist()
        accumulator.set_drifts(drifts_list, hdgs)

        accumulator.dist(return_pdf=False)

    if 'ndt' in params:
        if ('sigma_ndt' not in params) or (params['sigma_ndt'] == 0):
            params['sigma_ndt'] = 1e-100

        ndt_dist = norm.pdf(orig_tvec, loc=params['ndt'], scale=params['sigma_ndt'])

        if not use_signed_drifts:
            hdgs = hdgs[hdgs >= 0]

        for h in range(len(hdgs)):
            rt_dist = np.squeeze(accumulator.rt_dist[h, :])
            rt_dist = np.clip(rt_dist, np.finfo(np.float64).eps, a_max=None)
            rt_dist = convolve(rt_dist, ndt_dist / ndt_dist.sum())
            rt_dist = rt_dist[:len(accumulator.tvec)]
            rt_dist /= rt_dist.sum()  # renormalize

            accumulator.rt_dist[h, :] = rt_dist

    return accumulator, orig_tvec

## % ----------------------------------------------------------------
## HELPER FUNCTIONS


def get_stim_urgs(tvec: np.ndarray = None, skew_params: Optional[tuple] = None,
                 pos: Optional[np.ndarray] = None) -> np.ndarray:

    if pos is None:
        ampl = 0.16

        # pos = norm.cdf(tvec, 0.9, 0.3) * ampl
        if skew_params is None:
            pos = skewnorm.cdf(tvec, 2, 0.8, 0.4) * ampl  # emulate tf
        else:
            pos = skewnorm.cdf(tvec, **skew_params) * ampl

    vel = np.gradient(pos)
    acc = np.gradient(vel)

    vel /= vel.max()
    acc /= acc.max()

    return vel, acc


def set_params_list(params: np.ndarray, x0: np.ndarray,
                    fixed: np.ndarray = None) -> np.ndarray:
    """
    Set the parameter list using current iteration.

    Replaces parameter values with initial set value where fixed is true
    """
    if x0 is not None and (fixed is not None and fixed.sum() > 0):
        assert x0.shape == params.shape == fixed.shape, "x0 and fixed must match x in shape"
        params = np.array([q if is_fixed else p for p, q, is_fixed in zip(params, x0, fixed)])
    else:
        raise ValueError("initial values x0, or boolean fixed mask not provided")

    return params


def get_params_array_from_dict(params: dict, param_keys: list = None) -> np.ndarray:
    """Extract an array of parameters from the ordered keyword dictionary"""
    if param_keys is None:
        param_keys = ['kmult', 'bound', 'ndt', 'sigma_ndt']
        if 'theta' in params:
            param_keys = ['kmult', 'bound', 'ndt', 'sigma_ndt', 'theta', 'alpha']

        params = OrderedDict([(key, params[key]) for key in param_keys])

    values_list = []
    for value in params.values():
        if isinstance(value, (list, np.ndarray)):
            values_list.extend(value)
        else:
            values_list.append(value)

    return np.array(values_list)


def set_params_dict_from_array(params_array: np.ndarray, ref_dict: dict):
    """Insert an array of parameters back into a dictionary according to the set-up in ref_dict."""
    params_dict = OrderedDict()
    current_index = 0
    for key, value in ref_dict.items():
        if isinstance(value, list):
            value_length = len(value)
            params_dict[key] = params_array[current_index:current_index + value_length]
        elif isinstance(value, np.ndarray):
            value_length = value.shape[0]
            params_dict[key] = params_array[current_index:current_index + value_length]
        else:
            value_length = 1
            params_dict[key] = params_array[current_index]
        current_index += value_length

    return params_dict


## % ----------------------------------------------------------------
## PRIVATE FUNCTIONS

def _margconds_from_intersection(prob_ab, prob_a):
    """
    :param prob_ab: joint probability of A and B
    :param prob_a: marginal probability of A
    :return: 
        a_given_b: conditional probability of A given B
        prob_b: marginal probability of B
    """
    prob_a = prob_a.reshape(-1, 1)  # make it 2-D, for element-wise and matrix mults below

    # conditional probability
    b_given_a = (prob_ab / np.sum(prob_ab)) / prob_a

    prob_b = prob_a.T @ b_given_a

    if np.any(prob_b == 0):
        a_given_b = b_given_a * prob_a
    else:
        a_given_b = b_given_a * prob_a / prob_b

    return a_given_b, prob_b.flatten()


def _intersection_from_margconds(a_given_b, prob_a, prob_b):
    """
    Recover intersection of a and b using conditionals and marginals, according to Bayes theorem
    Essentially the inverse of margconds_from_intersection, and the two can be used together e.g.
    to update the intersections after adding a base_rate to prob_b

    :param a_given_b:
    :param prob_a:
    :param prob_b:
    :return:
        prob_ab: joint probability of A and B
        b_given_a
    """

    prob_ab = a_given_b * prob_b
    b_given_a = prob_ab / prob_a

    return prob_ab, b_given_a


