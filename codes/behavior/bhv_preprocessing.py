
import numpy as np
import pandas as pd
from pathlib import PurePath

# %% helper functions

def prop_se(x):
    return np.sqrt((np.mean(x)*(1-np.mean(x))) / len(x))


def cont_se(x):
    return np.std(x) / np.sqrt(len(x))


def gaus(x, ampl, mu, sigma, bsln):
    return ampl * np.exp(-(x-mu)**2 / (2*sigma**2)) + bsln


def prop_se_minmax(x):
    se = prop_se(x)
    return (np.mean(x)-se, np.mean(x)+se)


def drop_brfix(df, columns="choice"):
    return df.dropna(subset=columns, axis=0)

def drop_one_targs(df, columns=["oneTargChoice"]):
    return df.loc[(df[columns] == 0).all(axis=1), :]


def drop_outlierRTs_bygroup(func):
    def wrapper(df, grp_cols, *args, **kwargs):
        return df.groupby(grp_cols).apply(func, *args, **kwargs).reset_index(drop=True)
    return wrapper

@drop_outlierRTs_bygroup
def drop_outlierRTs(df, rt_range, metric="precomputed"):

    if metric == "precomputed":
        minRT, maxRT = rt_range

    elif metric == "stdev":
        rt_std = df['RT'].std()
        rt_mu = df['RT'].mean()
        minRT = np.max(rt_mu - rt_range*rt_std, 0)
        maxRT = rt_mu + rt_range*rt_std

    elif metric == "percentile":
        prcRT = np.percentile(df['RT'], rt_range)
        minRT, maxRT = prcRT[0], prcRT[1]

    return df.loc[(df['RT'] > minRT) & (df['RT'] <= maxRT), :]


def bin_conditions(df, columns, bin_ranges, bin_labels):
    for c, col in enumerate(columns):
        df[col] = pd.cut(df[col], bins=bin_ranges[c], labels=bin_labels[c])
    return df

def drop_columns(df, columns):
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df.drop(columns, axis=1)


def zero_one_choice(df):
    # to make 0..1, like PDW - more convenient for pRight calculations
    # but should type as category for other things!
    if np.max(df['choice']) == 2:
        df['choice'] -= 1
    return df

# %%

if __name__ == "__main__":

    subject = "lucio"
    folder = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/"
    filename = "lucio_20220512-20230605.csv"
    bhv_df = pd.read_csv(PurePath(folder, filename))

    hdg_bins = [-12, -8, -4, -2, -1, 1, 2, 4, 8, 12]
    hdg_labels = [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12]

    coh_bins = [0, 0.5, 1]
    coh_labels = [0.2, 0.7]

    bhv_df_clean = (bhv_df
                    .pipe(drop_brfix)
                    .pipe(drop_one_targs)
                    .pipe(zero_one_choice)
                    .pipe(drop_columns, columns=["TargMissed","confRT","insertTrial"])
                    .pipe(drop_outlierRTs, grp_cols=['modality'], rt_range=(0.25, 2))
                    .pipe(bin_conditions, column=['coherence','heading','delta'], bin_ranges=bins, bin_labels=labels)
                    )

    bhv_df_clean['heading'] = bhv_df_clean['heading'].astype('numeric')
    bhv_df_clean['modality'] = bhv_df_clean['modality'].astype('category')
