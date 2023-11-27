from pathlib import PurePath
import pandas as pd

def quick_load():

    data_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data'
    filename = 'lucio_20220512-20230602_neuralData.pkl'
    filename = PurePath(data_folder, 'lucio_neuro_datasets', filename)

    with open(filename, 'rb') as file:
        data = pd.read_pickle(file)

    pars = ['Tuning', 'Task']
    data = data.loc[data[pars].notna().all(axis=1) & data['is_good']]
    
    return data