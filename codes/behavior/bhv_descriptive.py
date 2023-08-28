# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit

from codes.behavior.bhv_preprocessing import *

# TODO functions list
# 1. extracting behavior from neural recording dataset
# 2. RT quantiles, plot vs confidence/accuracy
# 3. correct vs error metrics
# 4. decorator to loop any function over a grouping variable (i.e. day/block)
# 5. regression analyses


def behavior_means(df, conftask=2, by_conds=None):

    # if conftask == 1:
    #    df['PDW'] = df['conftask'] > 0.5

    # for confidence means, remove one-target confidence trials
    df_noOneTarg = df.loc[df['oneTargConf'] == 0]


    # pRight
    pRight = df.groupby(by=by_conds)['choice'].agg(
        [np.mean, prop_se]).dropna(axis=0).reset_index()

    # this achieves the same thing
    # pRight = pd.pivot_table(df, values='choice',
    #       index=['modality','coherence'], columns='heading',
    #       aggfunc=(np.mean,prop_se)).stack().reset_index()

    # repeat for pHigh
    pHigh = None
    if conftask == 1:
        pHigh = df_noOneTarg.groupby(by=by_conds)['conf'].agg(
            [np.mean, cont_se]).dropna(axis=0).reset_index()
    elif conftask == 2:
        pHigh = df_noOneTarg.groupby(by=by_conds)['PDW'].agg(
            [np.mean, prop_se]).dropna(axis=0).reset_index()


    # meanRT calculation - use continuous error func
    meanRT = df.groupby(by=by_conds)['RT'].agg(
        [np.mean, cont_se]).dropna(axis=0).reset_index()

    return pRight, pHigh, meanRT


def choice_logit_fit(df, num_hdgs: int = 200) -> pd.Series:

    hdgs = np.unique(df['heading']).reshape(-1, 1)
    xhdgs = np.linspace(np.min(hdgs), np.max(hdgs), num_hdgs).reshape(-1, 1)

    # logreg = smf.logit("choice ~ heading", data=bhv_df).fit()
    logreg = sm.Logit(df['choice'], sm.add_constant(df['heading'])).fit()
    yhat = logreg.predict(sm.add_constant(xhdgs))
    params = logreg.params['heading'], logreg.params['const']

    # alternatively, using sklearn
    # logreg = LogisticRegression().fit(
    #               df['heading'].to_numpy().reshape(-1, 1),
    #               df['choice'].to_numpy())
    # yhat = logreg.predict_proba(xhdgs)[:, 1]

    return pd.Series({'yhat': yhat, 'params': params})


def gauss_fit(df, p0: np.ndarray, y_var: str = 'choice', numhdgs: int = 200) -> pd.Series:

    hdgs = np.unique(df['heading']).reshape(-1, 1)
    xhdgs = np.linspace(np.min(hdgs), np.max(hdgs), numhdgs).reshape(-1, 1)

    if y_var == 'choice':

        probreg = sm.Probit(df['choice'], sm.add_constant(df['heading'])).fit(
            start_params=p0)
        yhat = probreg.predict(sm.add_constant(xhdgs))
        params = probreg.params['heading'], probreg.params['const']

    elif y_var == 'PDW':
        params, pcov = curve_fit(gaus, xdata=df['heading'], ydata=1-df['PDW'], p0=p0)
        yhat = 1 - gaus(xhdgs, *params).flatten()

    elif y_var == 'RT':
        params, pcov = curve_fit(gaus, xdata=df['heading'], ydata=df['RT'], p0=p0)
        yhat = gaus(xhdgs, *params).flatten()

    elif y_var == 'correct':
        ...  # TODO

    # # To compute 1SD error on parameters,
    # perr = np.sqrt(np.diag(pcov))

    return pd.Series({'yhat': yhat, 'hdgs': xhdgs.flatten(), 'params': params})


def bhv_gauss_fits(bhv_df, p0, y_vars=('choice','PDW','RT'),
                   by_conds: str = 'modality', numhdgs: int = 200) -> dict:

    grp_bhv_df = bhv_df.groupby(by=by_conds)

    fit_results = {
        y_var: grp_bhv_df.apply(gauss_fit, p, y_var, numhdgs).dropna(axis=0).reset_index()
        for y_var, p in zip(y_vars, p0)
    }

    # explode hdgs and yhat vectors
    fit_results = {k: df.drop('params', axis=1).explode(['hdgs', 'yhat']) for k, df in fit_results.items()}

    return fit_results


# %%

def cue_weighting(fit_results):

    wves_emp = None
    wves_pred = None

    for res in fit_results.keys():
        ...
        # TODO calculate wves pred and wves emp for each of pRight, PDW, RT
        # TODO first fix the behavior_fit function to do by_delta as well...
    return wves_emp, wves_pred

# %%


def plot_behavior_means(data_obs, data_fit=None, col='coherence', hue='modality', hue_colors='krb',
                        labels: list = None):

    # TODO test that this is flexible enough to plot with hue = delta, or with hue = coherence and col as delta
    # TODO add in pCorrect
    # TODO plan out more general/alternative version which can handle results from model fitting
    # data fit should be optional (i.e. if not provided, just draw the lines between the observed data points

    if labels is None:
        labels = ['pRight', 'pHigh', 'mean RT (s)']

    hdgs = np.unique(data_obs['choice']['heading'])
    hues = np.unique(data_obs['choice'][hue])
    cols = np.unique(data_obs['choice'][col])

    fig, axs = plt.subplots(nrows=len(data_obs), ncols=len(cols), figsize=(5, 7))

    for yi, (y_var, df_obs) in enumerate(data_obs.items()):
        ln_stl = '' if y_var in data_fit else '-'
        yerr_label = 'cont_se' if y_var == 'RT' else 'prop_se'
        ylims = [0.5, 1.2] if y_var == 'RT' else [0, 1]

        for ic, c in enumerate(cols):
            for ih, h in enumerate(hues):
                hcol = hue_colors[ih]

                inds = (df_obs[col] == c) & (df_obs[hue] == h)

                if np.sum(inds):
                    temp_df = df_obs.loc[inds, :]

                    axs[yi][ic].errorbar('heading', y='mean', yerr=yerr_label, data=temp_df,
                                          marker='.', ms=5, mfc=hcol, mec=hcol, linestyle=ln_stl)

                    if y_var in data_fit:
                        df_fit = data_fit[y_var]
                        df_fit = df_fit.loc[(df_fit[col] == c) & (df_fit[hue] == h)]
                        axs[yi][ic].plot(df_fit['hdgs'], df_fit['yhat'], color=hcol)

            # TODO more cosmetic stuff
            axs[yi][ic].set_xticks(hdgs)
            axs[yi][ic].set_ylim(ylims)
            axs[yi][ic].set_ylabel(labels[yi])

            # seaborn alternative
            # axs[0][c] = sns.lineplot(x='heading', y='mean', hue=hue, errorbar=None,
            #               data=df_obs.loc[df_obs['coherence']==coh], palette=hue_colors,
            #               ax=axs[yi][c])

    plt.show()


# %%

def plot_RTq_decorator(RTq_func):
    def wrapper(*args, **kwargs):
        RTq = RTq_func(*args, **kwargs)
        sns.relplot(data=RTq, x=RTq[('RT', 'mean')], y=RTq[(kwargs['depvar'], 'mean')], hue='heading', col='modality', style='coherence', kind='line')
        return RTq
    return wrapper


@plot_RTq_decorator
def RTquantiles(df: pd.DataFrame, by_conds, q_conds=None, nq: int=5, depvar: str = 'PDW', use_abs_hdg=True):

    # TODO this doesn't look quite right at the moment...
    """

    :param df:
    :param by_conds: how to group trials
    :param q_conds: how to group lines (by default, by_conds[:-1])
    :param nq: number of quantiles
    :param depvar:
    :return:
    """
    if q_conds is None:
        q_conds = by_conds[:-1]

    if use_abs_hdg:
        df['heading'] = df['heading'].abs()

    # assign a quantile to each trial in the df, and store the mid of each quantile (for plotting)
    def calc_bin_edges(x, num_bins):
        q = np.arange(1, num_bins+1) / (num_bins+1)
        return np.concatenate(([0], q, [1]))

    df.loc[:, 'RTq'] = df.groupby(q_conds)['RT'].transform(lambda x: pd.qcut(x, calc_bin_edges(x, nq), labels=False))

    #qvals = df.groupby(q_conds)['RT'].transform(lambda x: pd.qcut(x, calc_bin_edges(x, nq)))
    #df.loc[:, 'qmid'] = qvals.apply(lambda x: x.mid)

    agg_funcs = {
        depvar: ['mean', prop_se, 'count'],
        'RT': ['mean', cont_se],
    }
    RTq = df.groupby(by_conds + ['RTq'])[[depvar, 'RT']].agg(agg_funcs).dropna(axis=0).reset_index()

    return RTq


def basic_behavior_analysis(subject, date_range):

    # should just pass in the actual file name?
    bhv_df_clean = data_cleanup(subject=subject, date_range=date_range)

    # create a single column for PDW, with oneTarg trials coded as 2
    bhv_df_clean["PDW_1targ"] = bhv_df_clean['PDW']
    bhv_df_clean.loc[bhv_df_clean['oneTargConf'] == 1, 'PDW_1targ'] = 2
    bhv_df_clean["PDW_1targ"] = bhv_df_clean['PDW_1targ'].astype("category")

    # df with one-target PDW trials removed
    bhv_df_noOneTarg = bhv_df_clean.loc[bhv_df_clean['oneTargConf'] == 0]
    #bhv_df_noOneTarg = bhv_df_clean.loc[(bhv_df_clean['oneTargConf'] == 0) & (bhv_df_clean['delta'] == 0)]

    by_conds = ['modality', 'coherence', 'heading']
    RTq = RTquantiles(bhv_df_noOneTarg, by_conds=by_conds, q_conds=by_conds, nq=5, depvar='PDW')

    split_by = ['modality', 'coherence', 'heading', 'delta']
    pRight, pHigh, meanRT = behavior_means(bhv_df_clean, conftask=2, by_conds=split_by)

    # gaussian fits
    split_by = ['modality', 'coherence', 'delta']
    p0 = [[0, 3], [0.1, 0, 3, 0.5], [0.1, 0, 3, 0.5]]
    fit_results = bhv_gauss_fits(bhv_df_noOneTarg, p0=p0, y_vars=('choice','PDW', 'RT'), by_conds=split_by)

    data_obs = {'choice': pRight, 'PDW': pHigh, 'RT': meanRT}
    plot_behavior_means(data_obs, fit_results, hue='modality', col='coherence')

    # regression analyses

    # 1.
    formula = 'choice ~ heading + C(modality) + heading*C(modality)'
    fit_acc = smf.logit(formula, data=bhv_df_clean).fit()
    print(fit_acc.summary())

    bhv_df_clean['abs_heading'] = np.abs(bhv_df_clean['heading'])
    formula = 'correct ~ abs_heading*C(PDW_1targ)'
    fit_p = smf.logit(formula, data=bhv_df_clean).fit()
    print(fit_p.summary())


if __name__ == "__main__":
    basic_behavior_analysis("lucio", (20220512, 20230605))
