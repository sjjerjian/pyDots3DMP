# %% imports

import numpy as np
import pandas as pd
from pathlib import Path, PurePath
import scipy.io as sio
import pickle as pkl
from NeuralDataClasses import Population, ksUnit, Unit

# %% build Population class

def build_rec_popn(subject, rec_date, rec_info, data, data_folder):
    """
    Parameters
    ----------
    subject : TYPE
        DESCRIPTION.
    rec_date : TYPE
        DESCRIPTION.
    rec_info : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    data_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    rec_popn : TYPE
        DESCRIPTION.

    """

    session = f"{subject}{rec_date}_{rec_info['rec_set']}"
    # filepath = PurePath(data_folder, session)

    # get task timing and conditions - 'events'
    events = {**data['events'], **data['pldaps']}
    # events = pd.DataFrame({k: pd.Series(v) for k, v in events.items()})
    events = pd.DataFrame.from_dict(events, orient='index')
    events = events.T.convert_dtypes(infer_objects=True)
    events['heading'].loc[np.abs(events['heading']) < 0.01] = 0

    # initialize the neural population
    rec_popn = Population(subject=subject.upper()[0], rec_date=rec_date,
                          session=session, chs=rec_info['chs'], events=events)

    # add all the metadata from rec_info
    for key in rec_info.keys():
        vars(rec_popn)[key] = rec_info[key]

    # add the units

    # if only one unit, data fields will not be lists
    if isinstance(data['units']['cluster_id'], int):

        spk_times = np.array(data['units']['spiketimes'])
        clus_group = data['units']['cluster_type']
        unit = ksUnit(spiketimes=spk_times, amps=np.empty(spk_times.shape),
                      clus_id=data['units']['cluster_id'],
                      clus_group=clus_group,
                      clus_label=get_cluster_label(clus_group),
                      channel=data['units']['ch'],
                      depth=data['units']['depth'],
                      rec_date=rec_popn.rec_date, rec_set=rec_popn.rec_set)
        rec_popn.units.append(unit)

    else:
        nUnits = data['units']['cluster_id'].size
        for u in range(nUnits):
            spk_times = np.array(data['units']['spiketimes'][u])
            clus_group = data['units']['cluster_type'][u]
            unit = ksUnit(spiketimes=spk_times,
                          amps=np.empty(spk_times.shape),
                          clus_id=data['units']['cluster_id'][u],
                          clus_group=clus_group,
                          clus_label=get_cluster_label(clus_group),
                          channel=data['units']['ch'][u],
                          depth=data['units']['depth'][u],
                          rec_date=rec_popn.rec_date, rec_set=rec_popn.rec_set)
            rec_popn.units.append(unit)


    # # read in cluster groups (from manual curation)
    # cgs = pd.read_csv(PurePath(filepath, 'cluster_group.tsv'), sep='\t')
    
    # # read cluster info
    # clus_info = pd.read_csv(PurePath(filepath, 'cluster_info.tsv'), sep='\t')
    
    # ss = np.squeeze(np.load(PurePath(filepath, 'spike_times.npy')))
    # sg = np.squeeze(np.load(PurePath(filepath, 'spike_clusters.npy')))
    # st = np.squeeze(np.load(PurePath(filepath, 'spike_templates.npy')))
    # sa = np.squeeze(np.load(PurePath(filepath, 'amplitudes.npy')))

    # only go through clusters in this group of chs (i.e. one probe/area)
    # these_clus_ids = clus_info.loc[clus_info['ch'].isin(rec_info['chs']),
    #                                'cluster_id']
    # cgs = cgs.loc[cgs['cluster_id'].isin(these_clus_ids) &
    #               cgs['group'].isin(groups2keep), :]

    # for clus in cgs.itertuples():

    #     # get info for this cluster
    #     clus_id = clus.cluster_id
    #     unit_info = clus_info[clus_info['cluster_id'] == clus_id].to_dict(
    #         'records')[0]

    return rec_popn, events

# %%


def get_cluster_label(clus_group, labels=['unsorted', 'mua', 'good', 'noise']):
    """

    Parameters
    ----------
    clus_group : str
        0, 1, 2, or 3
    labels : list of strings, optional
        kilosort label. The default is ['unsorted', 'mua', 'good', 'noise'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    try:
        return labels.index(clus_group)
    except ValueError:
        return None


# %%


if __name__ == '__main__':

    # TODO - clean this up...
    # subject = input("Enter subject:")
    subject = 'lucio'
    datapath = '/Volumes/homes/fetschlab/data/'
    data_folder = Path(datapath, subject, f'{subject}_neuro/')

    mat_data_file = 'lucio_20220512-20230411_neuralData.mat'
    local_folder = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data/'

    m = sio.loadmat(PurePath(local_folder, 'lucio_neuro_datasets',
                             mat_data_file),
                    simplify_cells=True)
    data = m['dataStruct']

    # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    # rec_dates = [f.parts[-1] for f in data_folder.iterdir()
    #             if f.is_dir() and f.parts[-1][0:2] == '20']

    rec_info_file = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/RecSessionInfo.xlsx'
    rec_info = pd.read_excel(rec_info_file, sheet_name=subject.lower())
    rec_info = rec_info.convert_dtypes(infer_objects=True)
    rec_info.columns = rec_info.columns.str.lower()

    rec_info['date'] = rec_info['date'].apply(lambda x:
                                              x.date().strftime('%Y%m%d'))

    rec_info.rename(columns={'mdi_depth_um': 'probe_depth',
                             'gt_depth_cm': 'gt_depth',
                             'pen_no_total': 'pen_num',
                             'brain_area': 'area'}, inplace=True)

    pars = ['Tuning', 'Task']
    par_labels = ['dots3DMPtuning', 'dots3DMP']

    rec_df = rec_info.copy(deep=True)
    rec_df[pars] = pd.NA

    for index, sess in enumerate(data):

        rec_date = sess['date']
        rec_set = sess['rec_set']

        print(rec_date, rec_set)

        rec_folder = PurePath(data_folder, rec_date)

        rec_sess_info = rec_info.iloc[index, :].to_dict()
        rec_sess_info['chs'] = np.arange(rec_sess_info['min_ch'],
                                         rec_sess_info['max_ch']+1)
        rec_sess_info['grid_xy'] = (rec_sess_info['grid_x'],
                                    rec_sess_info['grid_y'])

        if not rec_sess_info['is_good']:
            continue

        for p, par in enumerate(pars):

            if type(sess['data']) == dict and par_labels[p] in sess['data'].keys():

                rec_popn, events = build_rec_popn(subject, rec_date, rec_sess_info,
                                                  data=sess['data'][par_labels[p]],
                                                  data_folder=rec_folder)

                rec_df.loc[index, par] = rec_popn

    filename = PurePath(local_folder, f"{mat_data_file.split('.')[0]}.pkl")
    with open(filename, 'wb') as file:
        pkl.dump(rec_df, file)
