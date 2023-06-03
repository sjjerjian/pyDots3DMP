# %%

# TODO
# functions for:
# 1. extracting behavior from neural recording dataset
# 2. RT quantiles, plot vs confidence/accuracy
# 3. correct vs error metrics
# 4. decorator to loop any function over a grouping variable (i.e. day/block)
# 5. regression analyses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn.objects as so
from pathlib import PurePath

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
# from sklearn.linear_model import LogisticRegression

# %% helper functions


def prop_se(x):
    return np.sqrt((np.mean(x)*(1-np.mean(x))) / len(x))


def cont_se(x):
    return np.std(x) / np.sqrt(len(x))


def gaus(x, ampl, mu, sigma, bsln):
    return ampl * np.exp(-(x-mu)**2 / (2*sigma**2)) + bsln


def clean_behavior_df(bhv_df):

    # TODO add trialNum from new version of Matlab preproc

    # clean up dataframe columns and types
    bhv_df = bhv_df.loc[:, ~bhv_df.columns.str.startswith('Unnamed')]

    # convert datetime to actual pandas datetime
    # TODO this seems to be missing time...
    bhv_df['datetime'] = pd.to_datetime(
        bhv_df['datetime'], infer_datetime_format=True)

    bhv_df['modality'] = bhv_df['modality'].astype('category')

    if np.max(bhv_df['choice']) == 2:
        # to make 0..1, like PDW - more convenient for calculations
        bhv_df['choice'] -= 1

    # remove one-target choice
    bhv_df = bhv_df.loc[bhv_df['oneTargChoice'] == 0]

    # leaving oneTargConf in, as this might be useful for some analyses
    # and can be removed easily later if desired

    # TODO remove outlier RTs
    # hard limits, or based on things beyond percentile/SDs from mean range

    return bhv_df

# %%


def behavior_means(bhv_df, conftask=2, by_conds=['modality', 'coherence',
                                             'heading']):
    """

    Parameters
    ----------
    bhv_df : TYPE
        DESCRIPTION.
    conftask : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    pRight : TYPE
        DESCRIPTION.
    pHigh : TYPE
        DESCRIPTION.
    meanRT : TYPE
        DESCRIPTION.

    """

    # if conftask is 1, we need a 'PDW' proxy for splitting by confidence
    # if conftask == 1:
    #    bhv_df['PDW'] = bhv_df['conftask'] > 0.5

    # for confidence means, remove one-target confidence trials
    bhv_df_noOneTarg = bhv_df.loc[bhv_df['oneTargConf'] == 0]

    # pRight calculation
    pRight = bhv_df.groupby(by=by_conds)['choice'].agg(
        [np.mean, prop_se]).dropna(axis=0).reset_index()

    # pHigh calculation
    pHigh = None
    if conftask == 1:
        pHigh = bhv_df_noOneTarg.groupby(by=by_conds)['conf'].agg(
            [np.mean, cont_se]).dropna(axis=0).reset_index()
    elif conftask == 2:
        pHigh = bhv_df_noOneTarg.groupby(by=by_conds)['PDW'].agg(
            [np.mean, prop_se]).dropna(axis=0).reset_index()

    # this effectively achieves the same thing
    # pRight = pd.pivot_table(bhv_df, values='choice',
    #       index=['modality','coherence'], columns='heading',
    #       aggfunc=(np.mean,prop_se)).stack().reset_index()

    # meanRT calculation
    meanRT = bhv_df.groupby(by=by_conds)['RT'].agg(
        [np.mean, cont_se]).dropna(axis=0).reset_index()

    return pRight, pHigh, meanRT

# %% descriptive fit to basic curves (none, logistic, or gaussian)


def choice_logit_fit(bhv_df, numhdgs=200):

    hdgs = np.unique(bhv_df['heading']).reshape(-1, 1)
    xhdgs = np.linspace(np.min(hdgs), np.max(hdgs), numhdgs).reshape(-1, 1)

    # logreg = smf.logit("choice ~ heading", data=bhv_df).fit()
    logreg = sm.Logit(bhv_df['choice'], sm.add_constant(bhv_df['heading'])).fit()
    yhat = logreg.predict(sm.add_constant(xhdgs))
    params = logreg.params['heading'], logreg.params['const']

    # using sklearn
    # logreg = LogisticRegression().fit(
    #               bhv_df['heading'].to_numpy().reshape(-1, 1),
    #               bhv_df['choice'].to_numpy())
    # yhat = logreg.predict_proba(xhdgs)[:, 1]

    return pd.Series({'yhat': yhat, 'params': params})


def gauss_fit(bhv_df, y='choice', p0=[0.1, 0, 3, 0.5], numhdgs=200):

    hdgs = np.unique(bhv_df['heading']).reshape(-1, 1)
    xhdgs = np.linspace(np.min(hdgs), np.max(hdgs), numhdgs).reshape(-1, 1)

    if y == 'choice':

        probreg = sm.Probit(bhv_df['choice'], sm.add_constant(bhv_df['heading'])).fit(
            start_params=p0)
        yhat = probreg.predict(sm.add_constant(xhdgs))
        params = probreg.params['heading'], probreg.params['const']

    elif y == 'PDW':
        params, _ = curve_fit(gaus,
                              xdata=bhv_df['heading'], ydata=1-bhv_df['PDW'], p0=p0)
        yhat = 1 - gaus(xhdgs, *params)

    elif y == 'RT':
        params, _ = curve_fit(gaus,
                              xdata=bhv_df['heading'], ydata=bhv_df['RT'], p0=p0)
        yhat = gaus(xhdgs, *params)

    return pd.Series({'yhat': yhat, 'params': params})


def bhv_summary_fits(bhv_df, by_conds=['modality', 'coherence']):

    # TODO allow p0 to be passed in

    grp_bhv_df = bhv_df.groupby(by=by_conds)

    fit_pRight = grp_bhv_df.apply(gauss_fit, 'choice', [0, 3]
                              ).dropna(axis=0).reset_index()

    fit_PDW = grp_bhv_df.apply(gauss_fit, 'PDW', [0.1, 0, 3, 0.5]
                           ).dropna(axis=0).reset_index()

    fit_RT = grp_bhv_df.apply(gauss_fit, 'RT', [0.1, 0, 3, 0.5]
                          ).dropna(axis=0).reset_index()

    return fit_pRight, fit_PDW, fit_RT



# def behavior_fit(bhv_df, fitType='logistic', numhdgs=200):

#     if np.max(bhv_df['choice']) == 2:
#         bhv_df['choice'] -= 1

#     hdgs = np.unique(bhv_df['heading']).reshape(-1, 1)
#     xhdgs = np.linspace(np.min(hdgs), np.max(hdgs), numhdgs).reshape(-1, 1)

#     outlist = ['pRight', 'PDW', 'RT']
#     modnames = ['ves', 'vis', 'comb']
#     attr_dict = {'intercept_': [], 'coef_': [], 'prediction': []}
#     fit_results = dict(zip(outlist, [dict(zip(modnames,
#                                               [attr_dict for m in modnames]))
#                                      for outcome in outlist]))


#     # always do fits separately for each mod/coh
#     for modality in pRight['modality'].unique():
#         for c, coh in enumerate(bhv_df['coherence'].unique()):

#             if modality == 1:
#                 X = bhv_df.loc[bhv_df['modality'] == modality,
#                            ['heading', 'choice', 'PDW', 'RT']]
#             else:
#                 X = bhv_df.loc[(bhv_df['modality'] == modality) &
#                            (bhv_df['coherence'] == coh),
#                            ['heading', 'choice', 'PDW', 'RT']]

#             if fitType == 'logistic':

#                 # using scikit-learn package
#                 # logreg = LogisticRegression().fit(
#                 #               X['heading'].to_numpy().reshape(-1, 1),
#                 #               X['choice'].to_numpy())
#                 # yhat = logreg.predict_proba(xhdgs)[:, 1]
#                 # coef_, intercept_ = logreg.coef_, logreg.intercept_

#                 logreg = sm.Logit(X['choice'], sm.add_constant(X['heading'])).fit()
#                 yhat = logreg.predict(sm.add_constant(xhdgs))
#                 coef_, intercept_ = logreg.params['heading'], logreg.params['const']

#             elif fitType == 'gaussian':

#                 probreg = sm.Probit(X['choice'], sm.add_constant(X['heading'])).fit()
#                 yhat = probreg.predict(sm.add_constant(xhdgs))
#                 coef_, intercept_ = probreg.params['heading'], probreg.params['const']

#                 # fits for PDW and RT
#                 popt_PDW, _ = curve_fit(gaus, xdata=X['heading'],
#                                         ydata=1-X['PDW'], p0=[0.1, 0, 3, 0.5])

#                 popt_RT, _ = curve_fit(gaus, xdata=X['heading'],
#                                        ydata=X['RT'], p0=[0.1, 0, 3, 0.5])

#                 fitPDW = 1 - gaus(xhdgs, *popt_PDW)
#                 fitRT = gaus(xhdgs, *popt_RT)

#                 fit_results['PDW'][modnames[modality-1]]['prediction'].append(fitPDW)
#                 fit_results['PDW'][modnames[modality-1]]['coef_'].append(popt_PDW)

#                 fit_results['RT'][modnames[modality-1]]['prediction'].append(fitRT)
#                 fit_results['RT'][modnames[modality-1]]['coef_'].append(popt_RT)

#             fit_results['pRight'][modnames[modality-1]]['prediction'].append(yhat)
#             fit_results['pRight'][modnames[modality-1]]['intercept_'].append(intercept_)
#             fit_results['pRight'][modnames[modality-1]]['coef_'].append(coef_)

#     return xhdgs, fit_results

# %%

def cue_weighting(fit_results):

    wves_emp = wves_pred = None

    for res in fit_results.keys():
        ...
        # TODO calculate wves pred and wves emp for each of pRight, PDW, RT
        # TODO first fix the behavior_fit function to do by_delta as well...
    return wves_emp, wves_pred

# %%


def plot_behavior_means(pRight, pHigh=None, meanRT=None):

    # (
    #     so.Plot(pRight, x='heading',y='mean', color='modality')
    #     .facet('coherence')
    #     .add(so.Line())
    # )

    condcols = ['k', 'r', 'b']
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))

    for modality in pRight['modality'].unique():

        for c, coh in enumerate(pRight['coherence'].unique()):

            if modality == 1:
                choice_mc = pRight[pRight['modality'] == modality]
                pdw_mc = pHigh[pHigh['modality'] == modality]
                rt_mc = meanRT[meanRT['modality'] == modality]
            else:
                choice_mc = pRight[(pRight['modality'] == modality) & (
                    pRight['coherence'] == coh)]
                pdw_mc = pHigh[(pHigh['modality'] == modality)
                               & (pHigh['coherence'] == coh)]
                rt_mc = meanRT[(meanRT['modality'] == modality)
                               & (meanRT['coherence'] == coh)]

            plt.sca(ax[0][c])
            ax[0][c].errorbar(choice_mc['heading'], choice_mc['mean'],
                              choice_mc['prop_se'], ecolor=condcols[modality-1])
            ax[0][c].set_xticks(choice_mc['heading'])

            plt.sca(ax[0][c])
            ax[1][c].errorbar(choice_mc['heading'],
                              pdw_mc['mean'], pdw_mc['prop_se'])
            ax[1][c].set_xticks(choice_mc['heading'])

            plt.sca(ax[0][c])
            ax[2][c].errorbar(choice_mc['heading'],
                              rt_mc['mean'], rt_mc['cont_se'])
            ax[2][c].set_xticks(choice_mc['heading'])
    plt.show()

# %%


def RTquantiles(bhv_df, nq=5, by_conds=['modality', 'coherence', 'heading'],
                depvar='PDW'):

    # TODO check this works and results make sense / compare with by hand calcs

    bhv_df['RTq'] = bhv_df.groupby(by_conds)['RT'].transform(
        lambda x: pd.qcut(x, nq, labels=False))

    # we also want to keep the RT (mean/median) of each quantile, for plotting
    quantile_vals = bhv_df.groupby(by_conds)['RT'].transform(
        lambda x: pd.qcut(x, nq))

    result = bhv_df.groupby(by_conds + ['RTq'])[depvar].mean().reset_index()

    return result

# %% define trial list


def dots3DMP_create_trial_list(hdgs, mods, cohs, deltas, nreps, shuff=True):

    # if shuff:
    #     np.random.seed(42)  # for reproducibility

    num_hdg_groups = any([1 in mods]) + any([2 in mods]) * len(cohs) + \
        any([3 in mods]) * len(cohs) * len(deltas)
    hdg = np.tile(hdgs, num_hdg_groups)

    coh = delta = modality = np.empty_like(hdg)

    if 1 in mods:
        coh[:len(hdgs)] = cohs[0]
        delta[:len(hdgs)] = 0
        modality[:len(hdgs)] = 1
        last = len(hdgs)
    else:
        last = 0

    # visual has to loop over cohs
    if 2 in mods:
        for c in range(len(cohs)):
            these = slice(last + c*len(hdgs), last + c*len(hdgs) + len(hdgs))
            coh[these] = cohs[c]
            delta[these] = 0
            modality[these] = 2
        last = these.stop

    # combined has to loop over cohs and deltas
    if 3 in mods:
        for c in range(len(cohs)):
            for d in range(len(deltas)):
                here = last + c*len(hdgs)*len(deltas) + d*len(hdgs)
                these = slice(here, here + len(hdgs))
                coh[these] = cohs[c]
                delta[these] = deltas[d]
                modality[these] = 3

    # Now replicate times nreps and shuffle (or not):
    condlist = np.column_stack((hdg, modality, coh, delta))
    trial_table = np.tile(condlist, (nreps, 1))
    ntrials = len(trial_table)

    if shuff:
        trial_table = trial_table[np.random.permutation(ntrials)]

    # TODO check this
    trial_table = np.stack(trial_table.T, axis=1)
    trial_table = pd.DataFrame(trial_table[:, [1, 2, 0, 3]],
                               columns=['modality', 'coherence',
                                        'heading', 'delta'])

    return trial_table, ntrials


# %% main

if __name__ == "__main__":

    subject = 'lucio'
    folder = '/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/'

    filename = PurePath(folder, 'lucio_20220512-20230222_clean.csv')
    bhv_df = pd.read_csv(filename)
    bhv_df = clean_behavior_df(bhv_df)

    # single PDW var, with oneTarg coded as 2
    bhv_df['PDW_1targ'] = bhv_df['PDW']
    bhv_df.loc[bhv_df['oneTargConf'] == 1, 'PDW_1targ'] = 2

    # drop delta trials if not using!

    split_by = ['modality', 'coherence', 'heading', 'PDW_1targ']
    pRight, pHigh, meanRT = behavior_means(bhv_df, conftask=2, by_conds=split_by)

    split_by = ['modality', 'coherence', 'PDW_1targ']
    fit_pRight, fit_PDW, fit_RT = bhv_summary_fits(bhv_df, by_conds=split_by)

    formula = 'choice ~ heading + C(modality) + heading*C(modality)'
    fit_acc = smf.logit(formula, data=bhv_df).fit()
    print(fit_acc.summary())

    formula = 'correct ~ PDW'
    fit_p = smf.logit(formula, data=bhv_df).fit()
    print(fit_p.summary())
    
    


    # mods = np.array([1, 2, 3])
    # cohs = np.array([1, 2])
    # hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
    # deltas = np.array([-3, 0, 3])
    # nreps = 1

    # trial_table, ntrials = \
    #     dots3DMP_create_trial_list(hdgs, mods, cohs, deltas,
    #                                nreps, shuff=False)