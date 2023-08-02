


# need a dataset with trial number as a variable

# function which takes a data dataframe, a list of M features (M>=1) to condition n-back on,
# optinally the factors of each feature that we want to condition on e.g. modality, [1 2 3] or [1, 2]
# or specify M features as dictionary with kv pairs
# and the n-back value (default 1), OR pre-defined df

# then add columns (or just create numpy arrays) with shifted M feature columns, loop over M feature combinations
# and assign all the rows which match criterion to a group


# func(df, ['modality', 'coherence'], [[1,2], [0.2]], n_back=2)

# will yield 3 groups in the outputted dataframe's new history_group column
# -1 or NaN for trials that don't fall in any history criterion
# 1 for trials that have n_back consec preceding trials of modality 1 with coherence 0.2
# 2 for trials that have n_back consec preceding trials of modality 2 with coherence 0.2

import numpy as np
import pandas as pd
from pathlib import PurePath
from statsmodels.tsa.tsatools import lagmat

from bhv_descriptive import clean_behavior_df

def consec_history_group(bhv_df, n_back=1, by_conds=['modality', 'coherence'], levels=(None, None)):

    features = lagmat(bhv_df[by_conds],
                      maxlag=2,
                      trim='forward',
                      original='ex',
                      use_pandas=True
                      )

    for cond in by_conds:
        cond_cols = [col for col in features.columns if cond in col]
        cond_features = features[cond_cols]

    return bhv_df_whistorygroup


if __name__ == '__main__':
    subject = 'lucio'
    folder = '/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/dataStructs/'

    filename = PurePath(folder, 'lucio_20220512-20230222_clean.csv')
    bhv_df = pd.read_csv(filename)
    bhv_df = clean_behavior_df(bhv_df)
    consec_history_group(bhv_df, by_conds=('modality', 'coherence'), levels=(None, None), n_back=1)
