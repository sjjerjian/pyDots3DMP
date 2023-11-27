import numpy as np
import pandas as pd

from typing import Optional, Sequence, Union

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, auc

from neural.rate_utils import condition_index


# TODO I think we should use LogisticRegressionCV at some point to identify the best hyperparameters for regularization
# the cross-validation in logistic_decoder is for actual evaluation

def logistic_regularization_fit():
    pass


def decode_outcome(f_rates: np.ndarray, condlist: pd.DataFrame, 
                 cond_groups: Optional[pd.DataFrame] = None, cond_cols: Optional[Sequence] = None,
                 outcome_col: Optional[str] = 'choice',
                 estimator = 'ROC',
                 pos_label = 1,  # ignored unless estimator == 'ROC'
                 decode_as_population=True, # ignored if estimator == 'ROC',
                 drop_nan_axis = 'units',
                 cv=10) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return 'decoding' scores given firing rates and binary outcome of interest
    - ROC analysis
    - or using an sklearn estimator e.g. logistic regression, SVM
    Calculate the outcome probability given firing rates using ROC analysis
    if outcome_col is 'choice', this will give the choice probabilities

    Args:
        f_rates (np.ndarray): single trial firing rates, units x trials x time
        condlist (pd.DataFrame): _description_
        cond_groups (pd.DataFrame, optional): _description_. Defaults to None.
        cond_cols (Sequence, optional): _description_. Defaults to None.
        outcome_col (str, optional): _description_. Defaults to 'choice'.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: decoding_result, unique conditions
    """
    
    if isinstance(estimator, str) and estimator.lower() == 'roc':
        decode_as_population = False # by necessity
        if isinstance(pos_label, int):
            pos_label = np.array([pos_label]*f_rates.shape[0])
        
        
    
    if cond_groups is not None:
        if cond_cols is None:
            cond_cols = cond_groups.columns[~cond_groups.columns.str.contains(outcome_col)]
            
        ic, nC, cg = condition_index(condlist[cond_cols], cond_groups[cond_cols].drop_duplicates())
    else:
        ic = np.zeros(f_rates.shape[1])
        nC, cg = 1, None

    # result is units x conditions x time if decoding individually
    if decode_as_population:
        outcome_score = np.full((nC, f_rates.shape[2]), np.nan)
    else:
        outcome_score = np.full((f_rates.shape[0], nC, f_rates.shape[2]), np.nan)
    
    for c in range(nC):
        if np.sum(ic == c):
            y_inds = condlist.loc[ic==c, outcome_col].to_numpy(dtype='int') 
            if np.max(y_inds) == 2:
                y_inds -= 1
            
            for t in range(f_rates.shape[2]):
            
                if isinstance(estimator, str) and estimator.lower()=='roc':
                    for u in range(f_rates.shape[0]):    
                        fr = f_rates[u, ic==c, t]

                        nan_idx = np.isnan(fr)
                        if nan_idx.sum() == len(fr) or len(np.unique(y_inds[~nan_idx]))==1 or np.isnan(pos_label[u]):
                            continue
                        
                        y_inds_good, fr_good = y_inds[~nan_idx], fr[~nan_idx]
                        
                        # flip left and right (assume all labels are 0 or 1)
                        if pos_label[u] == 0:
                            y_inds_good = 1 - y_inds_good

                        outcome_score[u, c, t] = roc_auc_score(y_inds_good, fr_good)
                        
                else:
                    if decode_as_population:
                        fr = f_rates[:, ic==c, t]
                        if drop_nan_axis == 'units':
                            nan_idx = np.isnan(fr).any(axis=1)
                            fr_good = fr[~nan_idx, :]
                            y_inds_good = y_inds
                            
                            if nan_idx.sum() == len(fr):
                                continue
                            
                        elif drop_nan_axis == 'trials':
                            nan_idx = np.isnan(fr).any(axis=0)
                            y_inds_good, fr_good = y_inds[~nan_idx], fr[:, ~nan_idx]

                            if nan_idx.sum() == len(fr) or len(np.unique(y_inds[~nan_idx]))==1:
                                continue
                            
                        cv_res = cross_validate(estimator, fr_good.T, y_inds_good, cv=cv)
                        outcome_score[c, t] = np.nanmean(cv_res['test_score'])
                        
                    else:
                        for u in range(f_rates.shape[0]): 
                            fr = f_rates[u, ic==c, t].reshape(-1, 1)

                            nan_idx = np.isnan(fr).any(axis=1)
                            if nan_idx.sum() == len(fr) or len(np.unique(y_inds[~nan_idx]))==1:
                                continue
                            y_inds_good, fr_good = y_inds[~nan_idx], fr[~nan_idx, :]
                            
                            cv_res = cross_validate(estimator, fr_good, y_inds_good, cv=cv)
                            outcome_score[u, c, t] = np.mean(cv_res['test_score'])
        
            
                               
    return outcome_score, cg



# alternative to in-built roc_auc_score...
def roc_auc_threshold(f_rates, labels, num_points=100):

    linspace_rates = np.linspace(np.min(f_rates), np.min(f_rates), num_points)
    
    # Initialize arrays to store false positive rates (FPR) and true positive rates (TPR)
    fpr = np.zeros(num_points)
    tpr = np.zeros(num_points)
    
    for i, threshold in enumerate(linspace_rates):
        # Calculate the ROC curve for the current threshold
        fpr[i], tpr[i], _ = roc_curve(labels, (f_rates >= threshold).astype(int))
    
    # Calculate the ROC AUC using the trapezoidal rule
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc