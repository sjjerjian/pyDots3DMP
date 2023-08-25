import numpy as np
import pandas as pd
from pathlib import Path, PurePath
import scipy.io as sio
import pickle as pkl
from codes.NeuralDataClasses import Population, ksUnit, PseudoPop

# %% build Population class


def build_rec_popn(subject: str, rec_date, rec_info, data) -> Population:

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
        unit = ksUnit(spiketimes=spk_times, amps=np.empty(spk_times.shape),
                      clus_id=data['units']['cluster_id'],
                      clus_group=data['units']['cluster_type'],
                      clus_label=data['units']['cluster_labels'],
                      channel=data['units']['ch'],
                      depth=data['units']['depth'],
                      rec_date=rec_popn.rec_date, rec_set=rec_popn.rec_set)
        rec_popn.units.append(unit)

    else:
        nUnits = data['units']['cluster_id'].size
        for u in range(nUnits):
            spk_times = np.array(data['units']['spiketimes'][u])
            unit = ksUnit(spiketimes=spk_times,
                          amps=np.empty(spk_times.shape),
                          clus_id=data['units']['cluster_id'][u],
                          clus_group=data['units']['cluster_type'][u],
                          clus_label=data['units']['cluster_labels'][u],
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


def build_pseudopop(fr_list, conds_dfs, unitlabels, tvecs=None, areas=None, subject=None):

    # conds_dfs = [df.assign(trialNum=np.arange(len(df))) for df in conds_dfs]

    # stack firing rates, along unit axis, with insertion on time axis according to t_idx
    num_units = np.array([x[0].shape[0] for x in fr_list])
    max_trs = max(list(map(len, list(conds_dfs))))

    stacked_frs = []

    # to stack all frs with time-resolutions preserved, make a single unique time vector (t_unq)
    # and insert each population matrix according to how its tvec lines up with t_unq
    # This is necessary in case of "dynamic" start/end references, different sessions might have different lengths
    #
    # only do it if tvecs is specified, otherwise the code returns interval averages

    t_unq, t_idx = [], []
    for j in range(len(fr_list[0])):

        u_pos = 0
        if tvecs is not None:  # or fr_list[0][0].ndim == 2:  # not sure why this was here...
            concat_tvecs = [tvecs[i][j] for i in range(len(tvecs))]
            t_unq.append(np.unique(np.concatenate(concat_tvecs)))
            t_idx.append([np.ravel(np.where(np.isin(t, t_unq[j]))) for t in concat_tvecs])

            stacked_frs.append(np.full([num_units.sum(), max_trs, len(t_unq[j])], np.nan))
            for ss in range(len(fr_list)):
                stacked_frs[j][u_pos:u_pos+num_units[ss], 0:len(conds_dfs[ss]), t_idx[j][ss]] = fr_list[ss][j]
                u_pos = u_pos + num_units[ss]

        else:
            stacked_frs.append(np.full([num_units.sum(), max_trs], np.nan))
            for ss in range(len(fr_list)):
                stacked_frs[j][u_pos:u_pos + num_units[ss], 0:len(conds_dfs[ss])] = fr_list[ss][j]
                u_pos = u_pos + num_units[ss]

    u_idx = np.array([i for i, n in enumerate(num_units) for _ in range(n)])
    area = [ar for fr, ar in zip(fr_list, areas) for _ in range(fr[0].shape[0])]

    # stacked_conds = [conds_dfs[u] for u in u_idx]

    return PseudoPop(
        subject=subject,
        firing_rates=stacked_frs,
        timestamps=t_unq,
        conds=conds_dfs,
        clus_group=np.hstack(unitlabels),
        area=area,
        unit_session=u_idx,
    )


def get_cluster_label(clus_group: int, labels=None) -> str:

    if labels is None:
        labels = ['unsorted', 'mua', 'good', 'noise']

    try:
        return labels.index(clus_group)
    except ValueError:
        return None


# %% MAIN

def create_population_data(datafile: str, info_file: str, save_folder='.') -> pd.DataFrame:

    # Load the matlab datastruct
    m = sio.loadmat(PurePath(save_folder, datafile), simplify_cells=True)
    data = m['dataStruct']

    # and load the Excel info sheet
    rec_info = pd.read_excel(info_file, sheet_name=subject.lower())
    rec_info = rec_info.convert_dtypes(infer_objects=True)
    rec_info.columns = rec_info.columns.str.lower()

    rec_info['date'] = rec_info['date'].apply(lambda x: x.date().strftime('%Y%m%d'))

    rec_info.rename(columns={'mdi_depth_um': 'probe_depth',
                             'gt_depth_cm': 'gt_depth',
                             'pen_no_total': 'pen_num',
                             'brain_area': 'area'}, inplace=True)

    pars = ['Tuning', 'Task']
    par_labels = ['dots3DMPtuning', 'dots3DMP']

    rec_df = rec_info.copy(deep=True)
    rec_df[pars] = pd.NA

    # loop over sessions, and paradigms, insert relevant data from info into df, and build population
    for index, sess in enumerate(data):

        rec_date = sess['date']
        rec_set = sess['rec_set']

        rec_sess_info = rec_info.iloc[index, :].to_dict()
        rec_sess_info['chs'] = np.arange(rec_sess_info['min_ch'],
                                         rec_sess_info['max_ch']+1)
        rec_sess_info['grid_xy'] = (rec_sess_info['grid_x'],
                                    rec_sess_info['grid_y'])

        if not rec_sess_info['is_good']:
            continue

        print(rec_date, rec_set)

        for p, par in enumerate(pars):

            if type(sess['data']) == dict and \
                    par_labels[p] in sess['data'].keys():

                rec_popn, events = build_rec_popn(
                    subject, rec_date, rec_sess_info, data=sess['data'][par_labels[p]]
                )
                rec_df.loc[index, par] = rec_popn

    if save_folder is not None:
        filename = PurePath(save_folder, f"{datafile.split('.')[0]}.pkl")
        with open(filename, 'wb') as file:
            pkl.dump(rec_df, file)

    return rec_df


if __name__ == '__main__':

    subject = "lucio"
    save_folder = f"/Users/stevenjerjian/Desktop/FetschLab/Analysis/data/{subject}_neuro_datasets"

    file = "lucio_20220512-20230602_neuralData.mat"
    info_file = "/Users/stevenjerjian/Desktop/FetschLab/Analysis/info/RecSessionInfo.xlsx"   # TODO add sheet for Lucio

    create_population_data(file, info_file, save_folder)