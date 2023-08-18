
import numpy as np
import pandas as pd
import codes.moi_dtb as dtb
from codes.behavior.bhv_preprocessing import *
from codes.behavior.bhv_simulation import dots3DMP_create_trial_list
import time

from scipy.signal import convolve
from scipy.stats import norm
from collections import defaultdict
from copy import deepcopy
from pybads import BADS


def optim_decorator(loss_func):
    def wrapper(params, init_params, fixed, *args, **kwargs):

        params = set_params_fixed(params, init_params, fixed)

        start_time = time.time()
        loss_val, _, _ = loss_func(params, *args, **kwargs)
        end_time = time.time()

        print(f"Likelihood evaluation time: {end_time-start_time:.2f}s")
        return loss_val

    return wrapper


def set_params_fixed(params, x0, fixed):

    if x0 is not None and (fixed is not None and fixed.sum() > 0):
        assert x0.shape == params.shape == fixed.shape, "x0 and fixed must match x in shape, if provided"
        params = np.array([q if is_fixed else p for p, q, is_fixed in zip(params, x0, fixed)])

    return params

@optim_decorator
def ddm_2d_objective(params: np.ndarray, data: pd.DataFrame,
                     accumulator: dtb.AccumulatorModelMOI = dtb.AccumulatorModelMOI(),
                     fit_options=None):


    if fit_options is None:
        fit_options = {'targets': ['choice', 'wager', 'RT'], 'RT_fittype': 'dist', 'fit_delta': False}

    # only calculate pdfs if fitting confidence variable
    return_pdf = ('wager' in fit_options['targets']) or ('multinomial' in fit_options['targets'])

    # TODO test/check these parameters, wrap with kwargs option to make it more explicit and better tested
    assert len(params) == 9, "Requires 9 parameters"

    kmult, B, theta_ves, theta_vis, theta_comb, alpha, tnd_ves, tnd_vis, tnd_comb = params
    theta = [theta_ves, theta_vis, theta_comb]
    tnd = [tnd_ves, tnd_vis, tnd_comb]

    cohs = np.unique(data['coherence'])
    mods = np.unique(data['modality'])
    hdgs = np.unique(data['heading'])
    deltas = [0]  # by default, don't fit cue conflict separately

    if fit_options['fit_delta']:
        deltas = np.unique(data['delta'])

    kvis = kmult * cohs.T
    kves = np.mean(kvis)


    model_data = data.loc[:, ['heading', 'coherence', 'modality', 'delta', 'choice', 'PDW', 'RT']]
    model_data[['p_right', 'p_high', 'model_rt', 'rt_llh']] = np.nan

    # TODO this is a placeholder, eventually should be acc, vel (+ additional urgency maybe)
    urg_ves, urg_vis = 1, 1

    # loop over conditions
    for m, mod in enumerate(mods):

        if mod == 1:
            abs_drifts = urg_ves * kves * np.sin(np.deg2rad(hdgs[hdgs >= 0]))
            signed_drifts = kves * np.sin(np.deg2rad(hdgs))

        # TODO if we collapse across kvis here, then how can pright vary for different cohs??
        # i think we collapse across kvis for log odds map, but not for choices/RT...
        elif mod == 2:
            abs_drifts = urg_vis * np.mean(kvis) * np.sin(np.deg2rad(hdgs[hdgs >= 0]))
            signed_drifts = np.mean(kvis) * np.sin(np.deg2rad(hdgs))

        elif mod == 3:

            if not fit_options['fit_delta']:
                abs_drifts = np.sqrt(urg_ves*kves**2 + urg_vis*np.mean(kvis)**2) * np.sin(np.deg2rad(hdgs[hdgs >= 0]))
                signed_drifts = np.sqrt(kves**2 + np.mean(kvis)**2) * np.sin(np.deg2rad(hdgs))

        # run the method of images dtb to extract pdfs, cdfs, and LPO
        # for fitting purposes, sensitivity (and drift) is fixed across coherences
        accumulator.set_drifts(list(abs_drifts))  # overwrite drift_rates
        accumulator.drift_labels = hdgs[hdgs >= 0]
        accumulator.dist(return_pdf=return_pdf)
        log_odds_above_threshold = accumulator.log_odds >= theta[m]
        #accumulator.plot()

        for c, coh in enumerate(cohs):

            for d, delta in enumerate(deltas):

                if delta != 0 and mod < 3:
                    continue

                if mod == 2:
                    abs_drifts = urg_vis * kvis[c] * np.sin(np.deg2rad(hdgs[hdgs >= 0]))
                    signed_drifts = kvis[c] * np.sin(np.deg2rad(hdgs))

                elif mod == 3:
                    w_ves = np.sqrt(kves ** 2 / (kves ** 2 + kvis[c] ** 2))
                    w_vis = np.sqrt(kvis[c] ** 2 / (kves ** 2 + kvis[c] ** 2))

                    # +ve delta means ves to the left, vis to the right
                    drift_ves = kves * np.sin(np.deg2rad(hdgs - delta / 2))
                    drift_vis = kves * np.sin(np.deg2rad(hdgs + delta / 2))

                    signed_drifts = w_ves * drift_ves + w_vis * drift_vis
                    # abs_drifts = np.abs(urg_ves*w_ves*drift_ves + urg_vis*w_vis*drift_vis)

                # update cdf to use cdf/deltas
                accumulator.set_drifts(list(abs_drifts))  # overwrite drift_rates
                accumulator.cdf()

                for h, hdg in enumerate(hdgs):

                    trial_index = (data['modality'] == mod) & (data['coherence'] == coh) & \
                                  (data['heading'] == hdg) & (data['delta'] == delta)


                    # for cue conflict, drift rates are not symmetrical around 0, so we need to use signed heading
                    if fit_options['fit_delta'] and mod == 3:
                        uh = h
                    else:
                        uh = np.abs(hdg) == hdgs[hdgs >= 0]

                    if return_pdf:
                        # select pdf for losing race, given correct or incorrect
                        pxt_up = np.squeeze(accumulator.up_lose_pdf[uh, :, :])
                        pxt_lo = np.squeeze(accumulator.lo_lose_pdf[uh, :, :])
                        total_p = np.sum(pxt_up + pxt_lo)
                        pxt_up /= total_p
                        pxt_lo /= total_p

                    # use signed drifts, to account for asymmetries in hdg angles around zero, from cue conflict
                    if signed_drifts[h] < 0:
                        p_right = 1 - accumulator.p_corr[uh].item()

                        if return_pdf:
                            # probabilities of four outcomes (intersections e.g. P(right AND high)), sum across grid
                            # for leftward headings, using pxt_lo (incorrect trials) for p_right
                            p_choice_and_wager = np.array(
                                [[np.sum(pxt_lo[log_odds_above_threshold]),  # pRight+High
                                 np.sum(pxt_lo[~log_odds_above_threshold])],  # pRight+Low
                                [np.sum(pxt_up[log_odds_above_threshold]),   # pLeft+High
                                 np.sum(pxt_up[~log_odds_above_threshold])]], # pLeft+Low
                            )
                    else:
                        p_right = accumulator.p_corr[uh].item()

                        if return_pdf:
                            p_choice_and_wager = np.array(
                                [[np.sum(pxt_up[log_odds_above_threshold]),  # pRight+High
                                  np.sum(pxt_up[~log_odds_above_threshold])],  # pRight+Low
                                 [np.sum(pxt_lo[log_odds_above_threshold]),  # pLeft+High
                                  np.sum(pxt_lo[~log_odds_above_threshold])]],  # pLeft+Low
                            )


                    if return_pdf:
                        p_choice_given_wager, p_wager = margconds_from_intersection(p_choice_and_wager,
                                                                                    np.array([p_right, 1-p_right]))

                        # factor in base-rate of low bets
                        p_wager += np.array([-alpha, alpha])*p_wager[0]
                        model_data.loc[trial_index, 'p_high'] = p_wager[0]

                    model_data.loc[trial_index, 'p_right'] = p_right

                    # draw sample model RTs based on full RT distribution pdf
                    # TODO convolve rt_dist with tnd_dist
                    rt_dist = np.squeeze(accumulator.rt_dist[uh, :])
                    rt_dist[-1] += (1-rt_dist.sum())  # any missing probability added to final bin...maybe should fix this earlier
                    model_data.loc[trial_index, 'model_rt'] = \
                        np.random.choice(accumulator.tvec, trial_index.sum(), replace=True, p=rt_dist)

                    # likelihood of observed RT is simply the probability at its value in the model RT distribution
                    actual_rts = model_data.loc[trial_index, 'RT'].values
                    dist_inds = [np.argmin(np.abs(accumulator.tvec - rt)) for rt in actual_rts]
                    model_data.loc[trial_index, 'rt_llh'] = rt_dist[dist_inds]

    # calculate log likelihoods
    model_llh = defaultdict(lambda: 0)

    model_data['p_right'].replace(to_replace=0, value=np.finfo(np.float64).tiny, inplace=True)
    model_data['p_right'].replace(to_replace=1, value=1-np.finfo(np.float64).tiny, inplace=True)
    model_data['rt_llh'].replace(to_replace=0, value=np.finfo(np.float64).tiny, inplace=True)

    model_llh['choice'] = np.sum(np.log(model_data.loc[model_data['choice'] == 1, 'p_right'])) + \
                          np.sum(np.log(1 - model_data.loc[model_data['choice'] == 0, 'p_right']))
    if return_pdf:
        model_data['p_high'].replace(to_replace=0, value=np.finfo(np.float64).tiny, inplace=True)
        model_data['p_high'].replace(to_replace=1, value=1-np.finfo(np.float64).tiny, inplace=True)

        model_llh['wager'] = np.sum(np.log(model_data.loc[model_data['PDW'] == 1, 'p_high'])) + \
                             np.sum(np.log(1 - model_data.loc[model_data['PDW'] == 0, 'p_high']))

    # RT likelihood straight from dataframe
    model_llh['RT'] = np.log(model_data['rt_llh']).sum()
    model_llh['RT'] /= 10  # seems to be a factor of 10 (roughly) larger than the other two.

    neg_llh = -np.sum(np.array([model_llh[v] for v in fit_options['targets'] if v in model_llh]))

    return neg_llh, model_llh, model_data


def margconds_from_intersection(prob_ab, prob_a):
    """
    Given joint probabilites of A and B, and marginal probability of A,
    returns marginal prob of B, and conditional of A given B, according to Bayes theorem

    :param prob_ab:
    :param prob_a:
    :return:
    """

    prob_a = prob_a.reshape(-1, 1)  # make it 2-D, so that the element-wise and matrix mults below work out

    # conditional probability
    b_given_a = (prob_ab/np.sum(prob_ab)) / prob_a

    prob_b = prob_a.T @ b_given_a
    a_given_b = b_given_a * prob_a / prob_b

    return a_given_b, prob_b.flatten()


def intersection_from_margconds(a_given_b, prob_a, prob_b):
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




def ddm_2d_generate_data(params, data=pd.DataFrame(),
                         accumulator: dtb.AccumulatorModelMOI = dtb.AccumulatorModelMOI(),
                         method='simulate',  # 'simulate', 'sample', or 'probability'
                         return_wager=True,
                         ):

    #use one function for generating data from model,
    # generate simulated trials (method='simulate'), return model probabilities ("
    # # only calculate pdfs if we ask for
    # return_pdf = any([t in options['targets'] for t in ['PDW', 'wager', 'multinomial']])

    #we need to be more flexible here, params should be a dict so we know what the keys are
    #
    # kmult, B, theta_ves, theta_vis, theta_comb, alpha, tnd_ves, tnd_vis, tnd_comb = params
    # theta = [theta_ves, theta_vis, theta_comb]
    # tnd = [tnd_ves, tnd_vis, tnd_comb]

    mods = np.unique(data['modality'])
    cohs = np.unique(data['coherence'])
    hdgs = np.unique(data['heading'])
    deltas = np.unique(data['delta'])

    model_data = data.loc[:, ['heading', 'coherence', 'modality', 'delta']]

    target_cols = ['choice', 'PDW', 'RT']
    model_data[target_cols] = np.nan

    # TODO this is a placeholder, eventually should be acc, vel (+ additional urgency maybe)
    urg_ves, urg_vis = 1, 1

    kvis = params['kmult'] * cohs.T
    kves = np.mean(kvis)  # for now, this means vis and ves do end up with the same log odds map
    accumulator.bound = params['B']

    temp_accumulator = deepcopy(accumulator)

    # ====== generate pdfs and log_odds for confidence
    # loop over modalities. marginalize over cohs, and ignore delta (assume confidence mapping is stable across these)

    if return_wager:
        sin_uhdgs = np.sin(np.deg2rad(hdgs[hdgs >= 0]))

        log_odds_maps = []
        for m, mod in enumerate(mods):

            if mod == 1:
                abs_drifts = urg_ves * kves * sin_uhdgs

            elif mod == 2:
                abs_drifts = urg_vis * np.mean(kvis) * sin_uhdgs

            elif mod == 3:
                abs_drifts = np.sqrt(urg_ves*kves**2 + urg_vis*np.mean(kvis)**2) * sin_uhdgs

            # run the method of images dtb to extract pdfs, cdfs, and LPO
            accumulator.set_drifts(list(abs_drifts))  # overwrite drift_rates
            accumulator.drift_labels = hdgs[hdgs >= 0]
            accumulator.dist(return_pdf=return_wager).log_posterior_odds()
            log_odds_maps.append(accumulator.log_odds)
        log_odds_above_threshold = [p >= theta for p, theta in zip(log_odds_maps, params['theta'])]

    # now to generate model results, loop over all conditions

    #TODO something is still breaking here, with p_choice having a 0 in it
    for m, mod in enumerate(mods):
        for c, coh in enumerate(cohs):
            for d, delta in enumerate(deltas):

                if (delta != 0 and mod < 3) or (c > 0 and mod > 1):
                    continue

                if mod == 1:
                    drifts = urg_ves * kves * np.sin(np.deg2rad(hdgs))

                if mod == 2:
                    drifts = urg_vis * kvis[c] * np.sin(np.deg2rad(hdgs))

                elif mod == 3:
                    w_ves = np.sqrt(kves ** 2 / (kves ** 2 + kvis[c] ** 2))
                    w_vis = np.sqrt(kvis[c] ** 2 / (kves ** 2 + kvis[c] ** 2))

                    # +ve delta means ves to the left, vis to the right
                    drift_ves = kves * np.sin(np.deg2rad(hdgs - delta / 2))
                    drift_vis = kves * np.sin(np.deg2rad(hdgs + delta / 2))

                    drifts = urg_ves * w_ves * drift_ves + urg_ves * w_vis * drift_vis

                # calculate cdf and pdfs using signed drift rates now

                condition_accumulator = temp_accumulator
                condition_accumulator.set_drifts(list(drifts))
                condition_accumulator.drift_labels = hdgs
                condition_accumulator.dist(return_pdf=True)
                print(drifts)

                for h, hdg in enumerate(hdgs):
                    print(mod, coh, hdg, delta)
                    trial_index = (data['modality'] == mod) & (data['coherence'] == coh) & \
                                  (data['heading'] == hdg) & (data['delta'] == delta)

                    if method == 'simulate':
                        for tr in range(trial_index.sum()):
                            dv = condition_accumulator.dv(drift=condition_accumulator.drift_rates[h])

                    else:
                        # otherwise, get model generated predictions, then either return those directly, or sample from them
                        ## Generate model choice, wager and RT for each trial in the dataset

                        p_right = condition_accumulator.p_corr[h].item()
                        p_choice = np.array([p_right, 1 - p_right])

                        print(hdg, p_right)
                        if p_right == 1 or p_right == 0:
                            print("stopping here")

                        # ====== WAGER ======
                        if return_wager:
                            # select pdf for losing race, given correct or incorrect
                            pxt_up = np.squeeze(condition_accumulator.up_lose_pdf[h, :, :])
                            pxt_lo = np.squeeze(condition_accumulator.lo_lose_pdf[h, :, :])
                            total_p = np.sum(pxt_up + pxt_lo)
                            pxt_up /= total_p
                            pxt_lo /= total_p

                            p_choice_and_wager = np.array(
                                [[np.sum(pxt_up[log_odds_above_threshold[m]]),  # pRight+High
                                  np.sum(pxt_up[~log_odds_above_threshold[m]])],  # pRight+Low
                                 [np.sum(pxt_lo[log_odds_above_threshold[m]]),  # pLeft+High
                                  np.sum(pxt_lo[~log_odds_above_threshold[m]])]],  # pLeft+Low
                            )

                            p_choice_given_wager, p_wager = margconds_from_intersection(p_choice_and_wager, p_choice)

                            # factor in base-rate of low bets
                            p_wager += np.array([-params['alpha'], params['alpha']]) * p_wager[0]

                            if np.any(np.isnan(p_wager)):
                                print("stopping here")

                            if method == 'sample':
                                model_data.loc[trial_index, 'PDW'] = np.random.choice([1, 0], trial_index.sum(),
                                                                                      replace=True, p=p_wager)
                            else:
                                model_data.loc[trial_index, 'PDW'] = p_wager[0]


                        # ====== CHOICE ======
                        if method == 'sample':
                            model_data.loc[trial_index, 'choice'] = np.random.choice([1, 0], trial_index.sum(),
                                                                                  replace=True, p=p_choice)
                        else:
                            model_data.loc[trial_index, 'choice'] = p_right

                        # ====== RT ======
                        # convolve model RT distribution with non-decision time
                        ndt_dist = norm.pdf(condition_accumulator.tvec, loc=params['ndt'][m], scale=params['sigma_ndt'])
                        rt_dist = np.squeeze(condition_accumulator.rt_dist[h, :])
                        rt_dist = convolve(rt_dist, ndt_dist/ndt_dist.sum())
                        rt_dist = rt_dist[:len(condition_accumulator.tvec)]
                        rt_dist /= rt_dist.sum()

                        #rt_dist[-1] += (1-rt_dist.sum())  # any missing probability added to final bin...maybe should fix this earlier

                        # given RT distribution, either store probability of observed RTs (if provided), or sample from distribution (for generating fake RTs)

                        try:
                            if method == 'sample':
                                model_data.loc[trial_index, 'RT'] = \
                                    np.random.choice(condition_accumulator.tvec, trial_index.sum(), replace=True, p=rt_dist)
                            else:
                                actual_rts = data.loc[trial_index, 'RT'].values
                                dist_inds = [np.argmin(np.abs(condition_accumulator.tvec - rt)) for rt in actual_rts]
                                model_data.loc[trial_index, 'RT'] = rt_dist[dist_inds]
                        except:
                            print("stopping here")

    return model_data, log_odds_maps






def main():

    data = data_cleanup("lucio", (20220512, 20230605))
    x0 = np.array([30, 1, 0.8, 0.7, 1.0, 0.05, 0.2, 0.4, 0.3])
    fixed = np.ones_like(x0)

    accum = dtb.AccumulatorModelMOI(tvec=np.arange(0, 2, 0.005), grid_vec=np.arange(-3, 0, 0.025))

    lb = np.array([10, 0.5, 0.5, 0.5, 0.5, 0, 0.15, 0.15, 0.15])
    ub = np.array([50, 2, 1.3, 1.3, 1.3, 0.1, 0.5, 0.5, 0.5])
    plb = np.array([20, 0.7, 0.5, 0.5, 0.5, 0, 0.15, 0.15, 0.15])
    pub = np.array([40, 1.5, 0.5, 0.5, 0.5, 0, 0.15, 0.15, 0.15])

    target = lambda params: ddm_2d_objective(params, x0, fixed, data, accum)
    bads = BADS(target, x0,)
    # optimize_result = bads.optimize()


if __name__ == '__main__':
    #main()

    mods = np.array([1, 2, 3])
    cohs = np.array([1, 2])
    hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
    deltas = np.array([-3, 0, 3])
    nreps = 1

    trial_table, ntrials = \
        dots3DMP_create_trial_list(hdgs, mods, cohs, deltas,
                                   nreps, shuff=False)

    accum = dtb.AccumulatorModelMOI(tvec=np.arange(0, 2, 0.005), grid_vec=np.arange(-3, 0, 0.025))

    params = {'kmult': 30, 'B': np.array([1, 1]), 'alpha': 0.05, 'theta': [0.8, 0.6, 0.7],
              'ndt': [0.3, 0.5, 0.4], 'sigma_ndt': 0.06}
    model_data_samp, _ = ddm_2d_generate_data(params, data=trial_table, accumulator=accum, method='sample')
    model_data_prob, _ = ddm_2d_generate_data(params, data=trial_table, accumulator=accum, method='probability')
    #model_data_sim, _ = ddm_2d_generate_data(params, data=trial_table, accumulator=accum, method='simulate')

    print("Done")
