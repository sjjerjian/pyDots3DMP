# %% imports

import numpy as np
import pandas as pd
from pathlib import Path, PurePath
import scipy.io as sio
from collections import defaultdict

import pickle as pkl
from neural.NeuralDataClasses import ksUnit, Population, RatePop
from neural.rate_utils import concat_aligned_rates, condition_averages

# %% ----------------------------------------------------------------
# build Population class from recorded data

def build_rec_popn(subject, rec_date, rec_info, data) -> Population:

    session = f"{subject}{rec_date}_{rec_info['rec_set']}"

    # get task timing and conditions - 'events'
    events = {**data['events'], **data['pldaps']}
    events = pd.DataFrame.from_dict(events, orient='index')
    events = events.T.convert_dtypes(infer_objects=True)
    events['heading'].loc[np.abs(events['heading']) < 0.01] = 0

    # initialize the neural population
    rec_popn = Population(subject=subject.upper(), rec_date=rec_date,
                          session=session, chs=rec_info['chs'], events=events)

    # add all the metadata from rec_info
    for key in rec_info.keys():
        vars(rec_popn)[key] = rec_info[key]

    # ===== add the units =====

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

    return rec_popn

# %% ----------------------------------------------------------------
# buld rate population (concatenated sessions)

def split_pseudopop_by_area(pseudopop: RatePop):
    
    areas = pseudopop.get_unique_areas()
        
    area_pseudopops = {k: [] for k in areas}
    for area in areas:
        area_pseudopops[area] = pseudopop.filter_units(pseudopop.area == area)
            
    return area_pseudopops
    

def split_pseudopop_decorator(create_pp_func):
    def wrapper(*args, **kwargs):
        pseudopop = create_pp_func(*args, **kwargs)        
        area_pseudopops = split_pseudopop_by_area(pseudopop)
        return area_pseudopops
    return wrapper

# @split_pseudopop_decorator  # this won't work with unstacked...
def build_rate_population(popn_dfs, tr_tab, t_params: dict, smooth_params: dict = None, 
                    event_time_groups: list = None, stacked=True, return_averaged=True) -> RatePop:
    
    
    num_sessions, num_alignments = len(popn_dfs), len(t_params['align_ev'])

    # --------------------------------
    # extract firing rates as unit x conditions x times, from each recording population's spiking activity
    
    fr_list, unitlabels, conds_dfs, tvecs, _ = zip(*popn_dfs.apply(
        lambda x: x.get_firing_rates(align_ev=t_params['align_ev'], trange=t_params['trange'],
                                     binsize=t_params['binsize'], sm_params=smooth_params,
                                     condlabels=tr_tab.columns)
        )
    )
    
    # --------------------------------
    # average trials within each condition specified in tr_tab
    if return_averaged:
        if t_params['binsize'] == 0:
            rates_cat, _, len_intervals = concat_aligned_rates(fr_list)
        else:
            rates_cat, _, len_intervals = concat_aligned_rates(fr_list, tvecs)

        cond_frs, cond_groups = [], []
        for f_in, cond in zip(rates_cat, conds_dfs):

            # avg firing rate over time across units, each cond, per session
            f_out, _, cg = condition_averages(f_in, cond, cond_groups=tr_tab)
            cond_frs.append(f_out)
            cond_groups.append(cg)

        # resplit by len_intervals, for pseudopop creation
        if len_intervals:
            fr_list = list(map(lambda f, x: np.split(f, x, axis=2)[:-1], cond_frs, len_intervals))
        else:
            fr_list = list(map(lambda f: np.split(f, f.shape[2], axis=2), cond_frs))
    
        # overwrite conds_dfs with unique conditions lists, instead of individual trials
        conds_dfs = cond_groups

    rel_event_times = None
    if 'other_ev' in t_params:
        
        cg_events = None
        if return_averaged:
            if event_time_groups:
                cg_events = tr_tab[event_time_groups].drop_duplicates()
            else:
                cg_events = tr_tab
                
        rel_event_times = popn_dfs.apply(
            lambda x: x.popn_rel_event_times(align=t_params['align_ev'],
                                             others=t_params['other_ev'],
                                             cond_groups=cg_events
                                            )
                                         )
        rel_event_times = list(rel_event_times)
        
    # conds_dfs = [df.assign(trialNum=np.arange(len(df))) for df in conds_dfs]

    # stack firing rates, along unit axis, with insertion on time axis according to t_idx
    num_units = np.array([x[0].shape[0] for x in fr_list])
    max_trs = max(list(map(len, list(conds_dfs))))


    # to stack all frs with time-resolutions preserved, make a single unique time vector (t_unq)
    # and insert each population fr matrix according to how its tvec lines up with t_unq
    # need to do this to handle variable start/end references, different sessions might have different lengths
    # e.g. motionOn - motionOff varies on each trial, and the limit across sessions will also vary
    # only do it if tvecs is specified, otherwise assume we are just using the interval averages

    stacked_frs, t_unq, t_idx = [], [], []

    # loop over alignments
    for j in range(num_alignments):

        u_pos = 0
        if t_params['binsize'] > 0:     #tvecs is not None  # or fr_list[0][0].ndim == 2

            if j==0:
                print("time vector provided, concatenating time-resolved firing rates into pseudo-population\n")

            concat_tvecs = [tvecs[i][j] for i in range(num_sessions)]
            t_unq.append(np.unique(np.concatenate(concat_tvecs)))
            t_idx.append([np.ravel(np.where(np.isin(t, t_unq[j]))) for t in concat_tvecs])

            stacked_frs.append(np.full([num_units.sum(), max_trs, len(t_unq[j])], np.nan))
            for sess in range(len(fr_list)):
                stacked_frs[j][u_pos:u_pos+num_units[sess], 0:len(conds_dfs[sess]), t_idx[j][sess]] = fr_list[sess][j]
                    u_pos += num_units[sess]

            else:
                # if j==0:
                #     print("concatenating interval average rates into pseudo-population\n")

                stacked_frs.append(np.full([num_units.sum(), max_trs], np.nan))
                for sess in range(len(fr_list)):
                stacked_frs[j][u_pos:u_pos+num_units[sess], 0:len(conds_dfs[sess])] = np.squeeze(fr_list[sess][j])
                u_pos += num_units[sess]

    # list of area, session number, and unit number within session, for each unit
    area = np.array([p.area for n, p in zip(num_units, popn_dfs) for _ in range(n)]) 
    u_idx = np.array([i for i, n in enumerate(num_units) for _ in range(n)])

    # make conditions list the same size as units (replicate conditions list for units within the same session)
    # stacked_conds = [conds_dfs[u] for u in u_idx]

    pseudo_pop = PseudoPop(
        subject=popn_dfs[0].subject,
        firing_rates=stacked_frs,
        timestamps=t_unq,
        psth_params=t_params,
        rel_events=rel_event_times,
        conds=conds_dfs,
        clus_group=np.hstack(unitlabels),
        area=area,
        unit_session=u_idx,
    )

    return pseudo_pop


# %% convert cluster group int into cluster label

def get_cluster_label(clus_group, labels=['unsorted', 'mua', 'good', 'noise']):

    try:
        return labels.index(clus_group)
    except ValueError:
        return None


# %% MAIN

def create_dataset(data_file: str, info_file: str, save_file: bool = True) -> pd.DataFrame:
    
    subject = Path(data_file).name[0:5]
    datapath = '/Volumes/homes/fetschlab/data/'
    data_folder = Path(datapath, subject, f'{subject}_neuro/')

    m = sio.loadmat(data_file, simplify_cells=True)
    data = m['dataStruct']

    # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    # rec_dates = [f.parts[-1] for f in data_folder.iterdir()
    #             if f.is_dir() and f.parts[-1][0:2] == '20']

    rec_info = pd.read_excel(info_file, sheet_name=subject.lower())
    rec_info = rec_info.convert_dtypes(infer_objects=True)
    rec_info.columns = rec_info.columns.str.lower()

    rec_info['date'] = rec_info['date'].apply(lambda x:
                                              x.date().strftime('%Y%m%d'))

    rec_info.rename(columns={'mdi_depth_um': 'probe_depth',
                             'gt_depth_cm': 'gt_depth',
                             'pen_no_total': 'pen_num',
                             'brain_area': 'area'}, inplace=True)

    par_names = ['tuning', 'task', 'rf', 'ves']
    par_labels = ['dots3DMPtuning', 'dots3DMP', 'RFmapping', 'VesMapping']

    # TODO keep all pars here...
    pars = ['Tuning', 'Task']

    rec_df = rec_info.copy(deep=True)
    rec_df[pars] = pd.NA

    for index, sess in enumerate(data):

        rec_date = sess['date']
        rec_set = sess['rec_set']
        rec_folder = PurePath(data_folder, rec_date)  # just for bookkeeping

        rec_sess_info = rec_info.iloc[index, :].to_dict()
        rec_sess_info['chs'] = np.arange(rec_sess_info['min_ch'], rec_sess_info['max_ch']+1)
        rec_sess_info['grid_xy'] = (rec_sess_info['grid_x'], rec_sess_info['grid_y'])

        if not rec_sess_info['is_good']:
            continue

        print(f"Adding {rec_date}, set {rec_set}, {rec_sess_info['area']}\n")

        for p, par in enumerate(pars):

            if type(sess['data']) == dict and \
                    par_labels[p] in sess['data'].keys():

                rec_popn = build_rec_popn(
                    subject, rec_date, rec_sess_info, data=sess['data'][par_labels[p]],
                    )

                rec_df.loc[index, par] = rec_popn

    if save_file:
        filename = f"{Path(data_file).stem}.pkl"
        with open(filename, 'wb') as file:
            pkl.dump(rec_df, file)
    
    return rec_df


if __name__ == '__main__':

    mat_data_file = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data/lucio_neuro_datasets/lucio_20220512-20230602_neuralData.mat'
    rec_info_file = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/info/RecSessionInfo.xlsx'

    create_dataset(mat_data_file, rec_info_file)
    
    

