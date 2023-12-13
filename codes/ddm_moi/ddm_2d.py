import numpy as np
import pandas as pd

from scipy.signal import convolve
from scipy.stats import norm, truncnorm, skewnorm
from collections import OrderedDict

from typing import Optional
from functools import wraps
from codetiming import Timer

from ddm_moi.Accumulator import AccumulatorModelMOI


def optim_decorator(loss_func):
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

        for key, val in params_dict.items():
            print(f"{key}: {val}\t")

        loss_val, llhs, model_data = loss_func(params_dict, *args, **kwargs)

        print(f"Total loss:{loss_val:.2f}")
        print({key: round(llhs[key], 2) for key in llhs})
        print('\n\n')

        if loss_val == np.inf:
            print(params_dict)
            raise ValueError("loss function evaluated to infinite")
        else:
            return loss_val

    return wrapper


@optim_decorator
def objective(params: dict, data: pd.DataFrame,
              outputs=None, llh_scaling=None, **gen_data_kwargs):

    if outputs is None:
        outputs = ['choice', 'PDW', 'RT']

    if llh_scaling is None:
        llh_scaling = np.ones(len(outputs))

    # only calculate pdfs if fitting confidence variable
    return_wager = 'PDW' in outputs

    # get model predictions (probabilistic) given parameters and trial conditions in data
    model_data, _, _ = generate_data(params=params, data=data,
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

    model_llh['RT'] = np.sum(np.log(model_data['RT']))

    # RT likelihood, straight from dataframe
    model_llh['RT'] = np.log(model_data['RT']).sum()

    # sum the individual log likelihoods, if included in outputs, and after scaling them according to llh_scaling
    neg_llh = -np.sum(
        np.array([model_llh[v]*w for v, w in zip(outputs, llh_scaling)
                  if v in model_llh]))

    return neg_llh, model_llh, model_data



@Timer(name="ddm_run_timer")
def generate_data(params: dict, data: pd.DataFrame(), accum_kw: dict,
                  method: str = 'simulate', rt_method: str = 'likelihood', save_dv: bool = False,
                  stim_scaling=True, return_wager: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Given an accumulator model and trial conditions, this function generates model outputs for behavioral variables.
    Setting method = 'simulate' will simulate decision variables on individual trials
    method = 'probability' will return the probabilities of rightward choice and high bet on each trial, and
       (assume there is already a dataset with actual observed choices, RTs, and wagers)
       return likelihood of the RTs
    :param params: parameters dictionary
    :param data: dataframe, containing at least modality, coherence, heading, delta
    :param accum_kw: dictionary keyword argumens for base accumulator instance
    :param method: 'sim[ulate]', 'samp[le]', or 'prob[ability]'
    :param rt_method: 'likelihood', 'mean', or 'peak'
    :param save_dv: boolean = False
    :param stim_scaling: boolean = True
    :param return_wager: True or False
    :return:
    """

    # TODO add wager_method options: 'log_odds', 'time', 'evidence'
    # TODO add cue weighting options: 'optimal', 'random', 'fixed'
    # TODO unit tests

    mods = np.unique(data['modality'])
    cohs = np.unique(data['coherence'])
    hdgs = np.unique(data['heading'])
    deltas = np.unique(data['delta'])

    model_data = data.loc[:, ['heading', 'coherence', 'modality', 'delta']]
    model_data[['choice', 'PDW', 'RT']] = np.nan

    # initialize accumulator
    accumulator = AccumulatorModelMOI(**accum_kw)
    accumulator.bound = params['bound']
    orig_tvec = accumulator.tvec
    orig_dt = np.diff(orig_tvec[:2]).item()

    full_dvs = None
    if save_dv:
        full_dvs = np.full((len(accumulator.tvec), 2, len(model_data)), np.nan)

    # set up sensitivity and other parameters
    if isinstance(params['ndt'], (int, float)):
        params['ndt'] = [params['ndt']] * len(mods)

    if (params['sigma_ndt'] == 0) or ('sigma_ndt' not in params):
        params['sigma_ndt'] = 1e-100

    if not stim_scaling:
        b_ves, b_vis = np.ones_like(accumulator.tvec), np.ones_like(accumulator.tvec)
        kmult_scaling = 10

    else:
        b_ves = get_stim_urg(tvec=accumulator.tvec, moment='acc')
        b_vis = get_stim_urg(tvec=accumulator.tvec, moment='vel')

        # new elapsed time is squared accumulated time course of sensitivity
        cumul_bves = np.cumsum((b_ves**2)/(b_ves**2).sum())
        cumul_bvis = np.cumsum((b_vis**2)/(b_vis**2).sum())

        kmult_scaling = 10

    # to be nx1 arrays
    b_ves, b_vis = b_ves.reshape(-1, 1), b_vis.reshape(-1, 1)

    # scale up, because we downweighted the input to parameters at a similar order of magnitude
    kmult = [k * kmult_scaling for k in params['kmult']]

    # set kves and kvis
    if isinstance(kmult, (int, float)) or len(kmult) == 1:
        kvis = kmult * cohs.T
        kves = np.mean(kvis)
    else:
        kves = kmult[0]
        if len(kmult) == 2:
            kvis = kmult[1] * cohs.T
        else:
            kvis = kmult[1:]

    # ====== generate pdfs and log_odds for confidence ======
    # looping over modalities
    # but marginalize over cohs, and ignore delta (assume confidence mapping is stable across these)
    # sensitivity is determined not just be k (internal sensivitity) but also b (external sensivitity)
    # i.e. time-dependent stimulus reliability

    log_odds_maps = []
    if return_wager:
        if isinstance(params['theta'], (int, float)):
            params['theta'] = [params['theta']] * len(mods)

        sin_uhdgs = np.sin(np.deg2rad(hdgs[hdgs >= 0]))

        for m, mod in enumerate(mods):

            if isinstance(params['bound'], (list, np.ndarray)) and len(params['bound']) == 3:
                accumulator.bound = params['bound'][m]

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
                    abs_drifts = np.cumsum(b_vis**2 * np.mean(kvis) * sin_uhdgs, axis=0)
                else:
                    abs_drifts = np.mean(kvis) * sin_uhdgs

            elif mod == 3:

                kves2, kvis2 = kves**2, np.mean(kvis)**2
                kcomb2 = kves2 + kvis2

                w_ves = np.sqrt(kves2 / kcomb2)
                w_vis = np.sqrt(kvis2 / kcomb2)

                # Eq 14
                if stim_scaling:
                    t_comb = (kves2 / kcomb2) * cumul_bves + (kvis2 / kcomb2) * cumul_bvis
                    accumulator.tvec = t_comb * orig_tvec[-1]

                    drift_ves = np.cumsum(b_ves**2 * kves * sin_uhdgs, axis=0)
                    drift_vis = np.cumsum(b_vis**2 * np.mean(kvis) * sin_uhdgs, axis=0)
                    abs_drifts = w_ves * drift_ves + w_vis * drift_vis
                else:
                    abs_drifts = np.sqrt(kcomb2).reshape(-1, 1) * sin_uhdgs

            # divide by the new passage of time if we applied the scaling, because we will re-multiply by it in the Accumulator Class logic,
            # and this will yield the *position* of the particle
            abs_drifts /= accumulator.tvec.reshape(-1, 1)

            if abs_drifts.ndim == 2:
                drifts_list = [abs_drifts[:, i:i+1] for i in range(abs_drifts.shape[1])]
            else:
                drifts_list = abs_drifts.tolist()

            # run the method of images dtb to extract pdfs, cdfs, and LPO
            accumulator.set_drifts(drifts_list, hdgs[hdgs >= 0])
            accumulator.dist(return_pdf=True).log_posterior_odds()
            log_odds_maps.append(accumulator.log_odds)

            # TODO this is where we would enforce different confidence mappings
            # simply by reassigning log_odds_maps / log_odds_above_threshold
            # if confidence is purely time dependent, then we would set a log odds threshold map that is all ones to the left of threshold, and zeros to the right
            # if purely evidence dependent, then set it based on grid_vec, 1 below, zero above

        log_odds_above_threshold = [p >= theta for p, theta in zip(log_odds_maps, params['theta'])]

    # ===== now to generate model results, loop over all conditions =====
    # here we used the signed drift rates, to allow for asymmetries e.g. because of cue conflict

    # print('generating model results')
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

                print(mod, coh, delta)

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

                    kves2, kvis2 = kves**2, kvis[c]**2
                    kcomb2 = kves2 + kvis2

                    w_ves = np.sqrt(kves2 / kcomb2)
                    w_vis = np.sqrt(kvis2 / kcomb2)

                    # Eq 14
                    if stim_scaling:
                        t_comb = (kves2 / kcomb2) * cumul_bves + (kvis2 / kcomb2) * cumul_bvis
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

                if method[:3] == 'sim':
                    drifts = np.gradient(drifts, axis=0)
                    drifts *= orig_dt

                    sigma_dv = params['sigma_dv'] * np.sqrt(orig_dt)
                    sigma_dv = np.array([sigma_dv, sigma_dv])

                elif stim_scaling:
                    # since we'll remultiply by tvec in the cdf/pdf code
                    drifts /= accumulator.tvec.reshape(-1, 1)


                # calculate cdf and pdfs using signed drift rates now
                if drifts.ndim == 2:
                    drifts_list = [drifts[:, i:i+1] for i in range(drifts.shape[1])]
                else:
                    drifts_list = drifts.tolist()
                accumulator.set_drifts(drifts_list, hdgs)

                # don't need to run this if simulating dv!
                if method[:3] != 'sim':
                    accumulator.dist(return_pdf=return_wager)

                for h, hdg in enumerate(hdgs):
                    trial_index = (data['modality'] == mod) & (data['coherence'] == coh) & \
                                  (data['heading'] == hdg) & (data['delta'] == delta)

                    if trial_index.sum() == 0:
                        continue

                    # simulate decision variables
                    if method[:3] == 'sim':
                        these_trials = np.where(trial_index)[0]

                        non_dec_time = truncnorm.rvs(ndt_min, ndt_max, loc=params['ndt'][m], scale=params['sigma_ndt'],
                                                     size=trial_index.sum())

                        for tr in range(trial_index.sum()):
                            # only do cumulative sum if not stim scaling!
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
                                # log odds map, take distance between winner and loser,
                                # then shift up as if winner did hit bound, so we can lookup log odds map

                            else:
                                # only one hits the bound
                                choice = np.where(bound_crossed)[0].item()
                                rt_ind = time_crossed[choice]
                                final_v = dv[rt_ind, choice ^ 1]

                            if return_wager:

                                grid_ind = np.argmin(np.abs(accumulator.grid_vec - final_v))
                                # log_odds = log_odds_maps[m][rt_ind, grid_ind]
                                wager = int(log_odds_above_threshold[m][rt_ind, grid_ind])  # lookup log odds thres
                                #wager *= (np.random.random() > params['alpha'])  # incorporate base-rate of low bets
                                model_data.loc[these_trials[tr], 'PDW'] = wager

                            # flip choice result so that left choices = 0, right choices = 1 in the output
                            model_data.loc[these_trials[tr], 'choice'] = choice ^ 1

                            # RT = decision time + non-decision time - motion onset latency
                            model_data.loc[these_trials[tr], 'RT'] = orig_tvec[rt_ind] + non_dec_time[tr] - 0.3

                            if save_dv:
                                full_dvs[:, :, these_trials[tr]] = dv

                    else:
                        # otherwise, get model generated predictions
                        # then either return those directly as probabilities
                        # or sample from them to generate 'trials'

                        p_right = accumulator.p_corr[h].item()
                        p_right = np.clip(p_right, 1e-10, 1-1e-10)

                        p_choice = np.array([p_right, 1 - p_right])

                        # ====== WAGER ======
                        if return_wager:
                            # select pdf for losing race, given correct or incorrect
                            pxt_up = np.squeeze(accumulator.up_lose_pdf[h, :, :])
                            pxt_lo = np.squeeze(accumulator.lo_lose_pdf[h, :, :])
                            total_p = np.sum(pxt_up + pxt_lo)
                            pxt_up /= total_p
                            pxt_lo /= total_p

                            p_choice_and_wager = np.array(
                                [[np.sum(pxt_up[log_odds_above_threshold[m]]),     # pRight+High
                                  np.sum(pxt_up[~log_odds_above_threshold[m]])],   # pRight+Low
                                 [np.sum(pxt_lo[log_odds_above_threshold[m]]),     # pLeft+High
                                  np.sum(pxt_lo[~log_odds_above_threshold[m]])]],  # pLeft+Low
                            )

                            # calculate p_wager using Bayes rule,
                            # then factor in base rate of low bets ("alpha")
                            p_choice_given_wager, p_wager = _margconds_from_intersection(p_choice_and_wager, p_choice)
                            p_wager += np.array([-params['alpha'], params['alpha']]) * p_wager[0]

                            if method[:4] == 'sample':
                                model_data.loc[trial_index, 'PDW'] = \
                                    np.random.choice([1, 0], trial_index.sum(), replace=True, p=p_wager)
                            else:
                                model_data.loc[trial_index, 'PDW'] = p_wager[0]

                        # ====== CHOICE ======
                        if method[:4] == 'samp':
                            model_data.loc[trial_index, 'choice'] = \
                                np.random.choice([1, 0], trial_index.sum(), replace=True, p=p_choice)
                        else:
                            model_data.loc[trial_index, 'choice'] = p_right

                        # ====== RT ======

                        # convolve model RT distribution with non-decision time

                        ndt_dist = norm.pdf(orig_tvec, loc=params['ndt'][m], scale=params['sigma_ndt'])
                        rt_dist = np.squeeze(accumulator.rt_dist[h, :])

                        rt_dist = np.clip(rt_dist, np.finfo(np.float64).eps, a_max=None)
                        rt_dist = convolve(rt_dist, ndt_dist / ndt_dist.sum())
                        rt_dist = rt_dist[:len(accumulator.tvec)]
                        rt_dist /= rt_dist.sum()  # renormalize

                        # given RT distribution,
                        #   sample from distribution (generating fake RTs) if method == 'sample'
                        #   or store probability of observed RTs, if method == 'probability'

                        # 12-2023 I think we need to sample from the orig_tvec here, not the changed tvec!

                        if method[:4] == 'samp' or rt_method[:4] == 'samp':
                            model_data.loc[trial_index, 'RT'] = \
                                np.random.choice(orig_tvec, trial_index.sum(), replace=True, p=rt_dist)
                            model_data.loc[trial_index, 'RT'] -= 0.3  # motion onset latency!!
                        else:
                            if rt_method == 'likelihood':
                                actual_rts = data.loc[trial_index, 'RT'].values + 0.3
                                dist_inds = [np.argmin(np.abs(orig_tvec - rt)) for rt in actual_rts]
                                rt_dist /= rt_dist.max()
                                model_data.loc[trial_index, 'RT'] = rt_dist[dist_inds]
                                # NOTE in this case we save the likelihood values, not an RT value!
                            else:
                                # check that data doesn't contain RT column?
                                if rt_method == 'mean':
                                    model_data.loc[trial_index, 'RT'] = (orig_tvec * rt_dist).sum() # expected value
                                elif rt_method == 'peak':
                                    model_data.loc[trial_index, 'RT'] = orig_tvec[np.argmax(rt_dist)]
                                model_data.loc[trial_index, 'RT'] -= 0.3  # motion onset latency!!

    # to avoid log(0) issues when doing log-likelihoods, replace zeros and ones
    if method[:4] == 'prob':
        model_data.loc[:, ['choice', 'PDW']] = model_data.loc[:, ['choice', 'PDW']].replace(to_replace=0, value=1e-10)
        model_data.loc[:, ['choice', 'PDW']] = model_data.loc[:, ['choice', 'PDW']].replace(to_replace=1, value=1 - 1e-10)

    return model_data, log_odds_maps, full_dvs


def _margconds_from_intersection(prob_ab, prob_a):
    """
    Given joint probabilites of A and B, and marginal probability of A.

    returns marginal prob of B, and conditional of A given B, according to Bayes theorem
    :param prob_ab:
    :param prob_a:
    :return:
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
    """

    prob_ab = a_given_b * prob_b
    b_given_a = prob_ab / prob_a

    return prob_ab, b_given_a


def get_stim_urg(tvec: np.ndarray = None, skew_params: Optional[tuple] = None,
                 pos: Optional[np.ndarray] = None, moment=1) -> np.ndarray:

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

    if moment == 1 or moment == 'vel':
        return vel

    elif moment == 2 or moment == 'acc':
        return acc


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

