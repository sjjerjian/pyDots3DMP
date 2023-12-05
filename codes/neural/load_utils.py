import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import PurePath
from typing import Optional

from behavior.preprocessing import dots3DMP_create_conditions
from neural.dots3DMP_build_dataset import build_rate_population, stack_rate_populations

def load_dataset(filename: str, pathname: Optional[str], pars: Optional[list]) -> pd.DataFrame:

    if pathname is None:
        pathname = PurePath()
    
    if pars is None:
        pars = ['Task']

    data = pd.read_pickle(PurePath(pathname, filename))
    data = data.loc[data[pars].notna().all(axis=1) & data['is_good']] 
    
    return data

def quick_load():
    data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data/lucio_neuro_datasets/'
    data = load_dataset('lucio_20220512-20230602_neuralData.pkl', data_folder, pars=['Tuning', 'Task'])
    
    return data

