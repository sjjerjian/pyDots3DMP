# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit

from functools import wraps
from behavior.preprocessing import prop_se, cont_se, gaus


# TODO functions list
# 1. extracting behavior from neural recording dataset
# 2. RT quantiles, plot vs confidence/accuracy
# 3. correct vs error metrics
# 4. decorator to loop any function over a grouping variable (i.e. day/block)
# 5. regression analyses


def behavior_means(df, by_conds='heading', long_format=True):

    # TODO fix this up 
    # p_right = _groupbyconds(df, by_conds, 'choice', prop_se)
    # p_high = _groupbyconds(df, by_conds, 'PDW', prop_se)
    # mean_rt = _groupbyconds(df, by_conds, 'RT', cont_se)
    # correct = _groupbyconds(df, by_conds, 'correct', prop_se)
    # return {'choice': p_right, 'PDW': p_high, 'RT': mean_rt, 'correct': p_correct}

    agg_funcs = {
        'choice': ['count', 'mean', prop_se],
        'PDW': ['count', 'mean', prop_se],
        'RT': ['count', 'mean', cont_se],
        'correct': ['count', 'mean', prop_se],
    }
    agg_funcs = {k:v for k,v in agg_funcs.items() if k in df.columns}

    output_vars = list(agg_funcs.keys())

    df_means = df.groupby(by=by_conds)[output_vars].agg(agg_funcs).dropna(axis=0).reset_index()
    df_means.columns = ['_'.join(col) if col[0] in output_vars else col[0] for col in df_means.columns]  # remove multi-level index

    if long_format:
        count_melt = df_means.melt(
            id_vars=by_conds, 
            value_vars=df_means.columns[df_means.columns.str.contains('count')],
            var_name='variable', value_name='count')

        means_melt = df_means.melt(
            id_vars=by_conds, 
            value_vars=df_means.columns[df_means.columns.str.contains('mean')],
            var_name='variable', value_name='mean')

        sems_melt = df_means.melt(
            id_vars=by_conds, 
            value_vars=df_means.columns[df_means.columns.str.contains('_se')],
            var_name='variable', value_name='se')

        df_means = count_melt.copy()
        df_means['mean'] = means_melt['mean']
        df_means['se'] = sems_melt['se']
        
        df_means['variable'] = df_means['variable'].apply(lambda x: x.split('_')[0])
    
    return df_means


def _groupbyconds(df, by_conds, data_col, errfcn):
    group_res = df.groupby(by=by_conds)[data_col].agg(['count', 'mean', errfcn]).dropna(axis=0).reset_index()
    group_res = group_res.rename(columns={errfcn.__name__: "sem"})
    return group_res


def replicate_ves(df):
    """
    replicate vestibular condition rows, for every coherence level
    """
    dup_ves = df.loc[df['modality']==1, :]
    ucohs = np.unique(df['coherence'])
    ucohs = ucohs[ucohs != np.unique(dup_ves['coherence'])]
    
    result_df = pd.concat([dup_ves.assign(coherence=coh) for coh in ucohs], ignore_index=True)
    result_df = pd.concat((df, result_df), ignore_index=True)
    
    return result_df


def logit_fit_choice_hdg(df, num_hdgs: int = 200) -> pd.Series:

    hdgs = np.unique(df['heading']).reshape(-1, 1)
    xhdgs = np.linspace(np.min(hdgs), np.max(hdgs), num_hdgs).reshape(-1, 1)

    # logreg = smf.logit("choice ~ heading", data=df).fit()  # using formula api
    logreg = sm.Logit(df['choice'], sm.add_constant(df['heading'])).fit()
    yhat = logreg.predict(sm.add_constant(xhdgs))
    params = logreg.params['heading'], logreg.params['const']

    # alternatively, using sklearn # TODO test
    # logreg = LogisticRegression().fit(df[['heading]], df['choice'])
    # yhat = logreg.predict_proba(xhdgs)[:, 1]

    return pd.Series({'yhat': yhat, 'params': params})


def gauss_fit_hdg(df, p0: np.ndarray, y_var: str = 'choice', numhdgs: int = 200) -> pd.Series:
    
    """
    gaussian fitting over headings to single set of data
    """
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

    return pd.Series({'hdgs': xhdgs.flatten(), 'yhat': yhat, 'params': params})


def gauss_fit_hdg_group(df, p0, y_vars: tuple = ('choice', 'PDW', 'RT'), by_conds = 'modality', numhdgs: int = 200) -> dict:
    """
    apply the gaussian
    """
    fit_results = {
        y_var: df.groupby(by=by_conds).apply(gauss_fit_hdg, p, y_var, numhdgs).dropna(axis=0).reset_index()
        for y_var, p in zip(y_vars, p0)
    }

    # explode hdgs and yhat vectors
    fit_results = {k: df.drop('params', axis=1).explode(['hdgs', 'yhat']) for k, df in fit_results.items()}

    return fit_results


def fit_results_to_dataframe(fit_results, by_conds = 'modality'):

    y_vars = list(fit_results.keys())

    by_conds.append('hdgs')
    fit_df = fit_results[y_vars[0]][by_conds].rename(columns={'hdgs': 'heading'})

    for label, df in fit_results.items():
        fit_df[label] = df['yhat']

    return fit_df


# %%

def cue_weighting(fit_results):

    wves_emp = None
    wves_pred = None

    for res in fit_results.keys():
        ...
        # TODO calculate wves pred and wves emp for each of pRight, PDW, RT
    return wves_emp, wves_pred

# %%


def plot_behavior_hdg(data_obs, data_fit, row: str = 'variable', col: str ='coherence',
                      hue: str = 'modality', palette=sns.color_palette(), **fig_kwargs):
    
    def _errbar_plot(x, y, yerr, **kwargs):
        plt.errorbar(x, y, yerr, **kwargs)

    # plot the empirical data points        
    g = sns.FacetGrid(data_obs, row=row, col=col, hue=hue,
                      palette=palette, sharey=False,
                      **fig_kwargs)
    g.map_dataframe(_errbar_plot, 'heading', 'mean', 'se',
            linestyle='', marker='.')
    
    # overlay the fit data as a line
    for ax_key, ax in g.axes_dict.items():
        ax_data = data_fit.loc[data_fit[col]==ax_key[1], :]
        sns.lineplot(data=ax_data, x='heading', y=ax_key[0],
                        hue=hue, ax=ax, palette=palette, legend=False)
        
        ax.set_title("")
        if 'choice' in ax_key:
            ax.set_title(f"coh = {ax_key[1]}")
            ax.set_ylim([0, 1])
            ax.set_ylabel('prop. right')
        elif 'PDW' in ax_key:
            ax.set_ylim([0, 1])
            ax.set_ylabel('prop. high')
        elif 'RT' in ax_key:
            # ax.set_ylim([0.5, 1.2])
            ax.set_ylabel('mean RT (s)')    

        xhdgs = np.unique(data_obs['heading'])
        ax.set_xticks(xhdgs)
        ax.set_xticklabels(xhdgs, rotation=40, ha='right')
    
    # TODO add legend back in
    plt.show()
    
    return g   

# %%

def plot_rtq(RTq_func):
    def wrapper(row=None, col='modality', hue='heading', *args, **kwargs):
        RTq = RTq_func(*args, **kwargs)
        
        #sns.relplot(data=RTq, x=RTq['RT']['mean'], y=RTq[kwargs['depvar']]['mean'], row=row, col=col, hue=hue, style=style, kind='line')
        
        g = sns.FacetGrid(RTq, row=row, col=col, hue=hue, aspect=1.5, height=4, sharex=False, sharey=False)

        # Iterate through each subplot and plot errorbars
        def plot_errorbars(data, **kwargs):
            plt.errorbar(x=data['RT']['mean'], y=data[kwargs['depvar']]['mean'],
                         xerr=data['RT']['cont_se'], yerr=data[kwargs['depvar']]['prop_se'],
                         marker='.', markersize=5, capsize=3)
        g.map_dataframe(plot_errorbars, **kwargs)
        
        # Customize labels and legend
        g.set_axis_labels('RT Mean', f"{kwargs['depvar']} Mean")
        g.add_legend()
        
        # Show the plot
        plt.show()
                
        return RTq
    return wrapper


#@plot_rtq
def RTquantiles(df: pd.DataFrame, by_conds, q_conds=None, nq: int=5, depvar: str = 'PDW', use_abs_hdg=True):

    """

    :param df:
    :param by_conds: how to group trials
    :param q_conds: how to group lines (by default, by_conds[:-1])
    :param nq: number of quantiles
    :param depvar:
    :return:
    """
    if q_conds is None:
        q_conds = by_conds

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
    RTq.columns = ['_'.join(col) if col[0]==depvar or col[0]=='RT' else col for col in RTq.columns]  # remove multi-level index

    return RTq

            