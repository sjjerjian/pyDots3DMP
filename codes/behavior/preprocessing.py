
import numpy as np
import pandas as pd
from pathlib import Path, PurePath

from itertools import product


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


def drop_brfix(df, columns="choice") -> pd.DataFrame:
    return df.dropna(subset=columns, axis=0)

def drop_one_targs(df, columns=["oneTargChoice"]) -> pd.DataFrame:
    return df.loc[(df[columns] == 0).all(axis=1), :]


def drop_outlierRTs_bygroup(func):
    def wrapper(df, grp_cols=None, *args, **kwargs):
        return df.groupby(grp_cols).apply(func, *args, **kwargs).reset_index(drop=True)
    return wrapper


#@drop_outlierRTs_bygroup
def drop_outlierRTs(df, rt_range, metric: str = "precomputed") -> pd.DataFrame:

    min_rt, max_rt = 0, np.inf

    if metric == "precomputed":
        min_rt, max_rt = rt_range

    elif metric == "stdev":
        rt_std = df['RT'].std()
        rt_mu = df['RT'].mean()
        min_rt = np.max(rt_mu - rt_range*rt_std, 0)
        max_rt = rt_mu + rt_range*rt_std

    elif metric == "percentile":
        prc_rt = np.percentile(df['RT'], rt_range)
        min_rt, max_rt = prc_rt[0], prc_rt[1]

    return df.loc[(df['RT'] > min_rt) & (df['RT'] <= max_rt), :]


def bin_conditions(df, bin_ranges, bin_labels) -> pd.DataFrame:
    for col, bins in bin_ranges.items():
        df[col] = pd.cut(df[col], bins=bins, labels=bin_labels[col])
        df[col] = df[col].astype('float')
    return df


def drop_columns(df, columns) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df.drop(columns, axis=1)


def zero_one_choice(df) -> pd.DataFrame:
    # to make 0..1, like PDW - more convenient for pRight calculations
    # but should type as category for other things!
    if np.max(df['choice']) == 2:
        df['choice'] -= 1
    return df


def data_cleanup(filename: str, drop_cols=None, save_file: bool = False) -> pd.DataFrame:

    # TODO add kwargs for drop and binning parameters below, currently hardcoded...
    # TODO add print statements to explain, allow user-inputs to specify what functions to use?

    folder = "/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/"
   
    clean_filename = Path(filename).stem + "_clean.csv"

    bhv_df = pd.read_csv(PurePath(folder, filename))

    bins = {
        'coherence': [0, 0.5, 1],
        'heading': [-14, -8, -4, -2, -1, 1, 2, 4, 8, 14],
        'delta': [-5, -2, 2, 5],
    }
    labels = {
        'coherence': [0.2, 0.7],
        'heading': [-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12],
        'delta': [-3, 0, 3],
    }

    if drop_cols is None:
        drop_cols = ["TargMissed", "confRT", "insertTrial", "filename", "subj", "trialNum",
                        "amountRewardLowConfOffered", "amountRewardHighConfOffered", "reward"]

    bhv_df_clean = (bhv_df
                    .pipe(drop_brfix, columns=['choice', 'PDW'])
                    .pipe(drop_one_targs)
                    .pipe(zero_one_choice)
                    .pipe(drop_columns, columns=drop_cols)
                    .pipe(drop_outlierRTs, rt_range=(0.25, 2))
                    .pipe(bin_conditions, bin_ranges=bins, bin_labels=labels)
                    )

    # force all ves to be low coherence, by convention
    bhv_df_clean.loc[bhv_df_clean['modality'] == 1, 'coherence'] = bhv_df_clean['coherence'].min()

    # convert modality column to category type
    bhv_df_clean['modality'] = bhv_df_clean['modality'].astype('category')

    if save_file:
        bhv_df_clean.to_csv(PurePath(folder, clean_filename))
    
    return bhv_df_clean


def format_onetargconf(df: pd.DataFrame, remove_one_targ: bool = True) -> pd.DataFrame:

    if remove_one_targ:
        # create new df with one-target PDW trials removed
        return df.loc[df['oneTargConf'] == 0]

    else:
        # create a single category type column for PDW, with oneTarg trials coded as "2"
        df["PDW_1targ"] = df['PDW']
        df.loc[df['oneTargConf'] == 1, 'PDW_1targ'] = 2
        df["PDW_1targ"] = df['PDW_1targ'].astype("category")

    return df


def dots3DMP_create_trial_list(hdgs: list, mods: list, cohs: list, deltas: list,
                               nreps: int = 1, shuff: bool = True) -> pd.DataFrame:

    if isinstance(shuff, int):
        np.random.seed(shuff)  # for reproducibility

    num_hdg_groups = any([1 in mods]) + any([2 in mods]) * len(cohs) + \
        any([3 in mods]) * len(cohs) * len(deltas)
    hdg = np.tile(hdgs, num_hdg_groups)

    coh = np.empty_like(hdg)
    modality = np.empty_like(hdg)
    delta = np.empty_like(hdg)

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
    condlist = np.column_stack((modality, coh, delta, hdg))
    trial_table = np.tile(condlist, (nreps, 1))
    ntrials = len(trial_table)

    if shuff:
        if isinstance(shuff, int):
            trial_table = trial_table[np.random.default_rng(shuff).permutation(ntrials)]
        else:
            trial_table = trial_table[np.random.permutation(ntrials)]

    trial_table = pd.DataFrame(trial_table, columns=['modality', 'coherence', 'delta', 'heading'])

    return trial_table


def add_trial_outcomes(trial_table: pd.DataFrame, outcomes: dict = None) -> pd.DataFrame:

    """Replicate a trial table of stimulus conditions N times to include possible trial outcomes
    e.g. choice and wager

    Returns:
        pd.DataFrame: final trial conditions list, with additional columns for trial outcomes
    """
    if outcomes is None:
        outcomes = {'choice': [0, 1], 'PDW': [0, 1], 'oneTargConf': [0]}

    combinations = list(product(*outcomes.values()))
    df = pd.DataFrame(combinations, columns=outcomes.keys())
    
    new_trial_table = trial_table.loc[trial_table.index.repeat(len(df))].reset_index(drop=True)
    df_rep = pd.concat([df] * len(trial_table), ignore_index=True)
    new_trial_table = pd.concat([new_trial_table, df_rep], axis=1)

    return new_trial_table


def dots3DMP_create_conditions(conds_dict: dict[str, list], cond_labels=None) -> pd.DataFrame:
    
    # TODO allow user to set stim and res_keys
    stim_keys = ['mods', 'cohs', 'deltas', 'hdgs']    
    res_keys = ['choice', 'correct', 'PDW', 'oneTargConf']
    
    ss_dict = {key: conds_dict.get(key, [0]) for key in stim_keys}
    conds = dots3DMP_create_trial_list(**ss_dict, nreps=1, shuff=False)
                
    rr_dict = {key: conds_dict.get(key, [0]) for key in res_keys}
    conds = conds.pipe(add_trial_outcomes, rr_dict)
    
    indices = [idx for idx, el in enumerate(stim_keys+res_keys)
               if el in conds_dict]
    
    conds = conds[conds.columns[indices]]
    
    if cond_labels:
        conds.columns = cond_labels

    return conds
    
    
    
    
                               
    
    
        
                               
    
    