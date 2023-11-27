# %% ===== imports =====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from behavior.preprocessing import data_cleanup, format_onetargconf
from behavior.descriptive import *


# %% ===== load data =====

# at some point, just load pre-cleaned file here

monkey = 'lucio'

if monkey == 'lucio':
    filename = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/lucio_20220512-20230605.csv"
else:
    ...

bhv_df = data_cleanup(filename)
bhv_df_noOneTarg = format_onetargconf(bhv_df, remove_one_targ=True)


# %% ===== PDW and Accuracy against RT quantiles ===== 

by_conds = ['modality', 'coherence', 'heading']
data_delta0 = bhv_df_noOneTarg.loc[bhv_df_noOneTarg['delta']==0, :]

RTquantiles(df=data_delta0, by_conds=by_conds, nq=5, depvar='PDW')


# %% ===== Psychometric and chronometric curves ===== 

by_conds = ['modality', 'coherence', 'heading', 'delta']
data_means = behavior_means(bhv_df_noOneTarg, by_conds=by_conds)

# gaussian fits
by_conds = ['modality', 'coherence', 'delta']
p0 = [[0, 3], [0.1, 0, 3, 0.5], [0.1, 0, 3, 0.5]]
fit_results = gauss_fit_hdg(bhv_df_noOneTarg, p0=p0, y_vars=('choice','PDW', 'RT'), by_conds=by_conds)

data_delta0 = data_means.loc[data_means['delta']==0, :]
plot_behavior_hdg(data_delta0, fit_results, hue='modality', col='coherence')

# %% ===== Regression Analyses ===== 

formula = 'choice ~ heading + C(modality) + heading*C(modality)'
fit_acc = smf.logit(formula, data=bhv_df_noOneTarg).fit()
print(fit_acc.summary())

bhv_df_noOneTarg['abs_heading'] = np.abs(bhv_df_noOneTarg['heading'])
formula = 'correct ~ abs_heading*C(PDW_1targ)'
fit_p = smf.logit(formula, data=bhv_df_noOneTarg).fit()
print(fit_p.summary())

