# %% imports

import numpy as np
import pandas as pd
from pathlib import Path, PurePath

import scipy.io as sio
import pickle as pkl

from typing import Optional, Union, Sequence

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
# build rate population

    """
    A RatePopulation object can contain data from a single ("simultaneous")
    or multiple ("pseudo-population") recording populations.

    Firing rates arrays are units x nconds/ntrials x time, there can be
    a list of arrays if spikes are aligned to multiple behavioural events.
    Firing rates can be interchanged between a list of alignments and a concatenated
    array using split_alignments and concat_alignments.

    A RatePopulation's firing rates can be single-trial based or trial-averaged.
    Single-trial based firing rates will have an associated conditions dataframe
    that is ntrials in length, and the trial-averaged RatePopulation conditions
    dataframe will be nconds in length.

    If multiple simultaneous populations are stacked to create a PseudoPopulation,
    and trials are averaged, there will be a single conditions dataframe, since all
    trial-averaged neural firing rates are associated with the same conditions list.

    If trials are not averaged, then a list of conditions dataframes will be kept,
    one for each individual session.

    TODO add methods/functions to move between these options after creation
    (stacked to unstacked, unstacked to stacked (DONE), and single-trials to averaged).

    """


def build_rate_population(popns: Union[Sequence, pd.Series], tr_table: pd.DataFrame, t_params: dict, smooth_params: dict = None,
                    event_time_groups: list = None, stacked=True, return_averaged=True) -> Union[RatePop, list]:
    """Build a RatePop of firing rates given user-specified parameters (for PSTH alignment, resolution, and task conditions)

    Args:
        popns (_type_): sequence of Population Objects containing spike times for simulataneously recorded units, and task event information
        tr_table (pd.DataFrame): dataframe of unique trial conditions
        t_params (dict): PSTH parameters - alignment event(s), time ranges, binsize, other event(s) for relative timing
        smooth_params (dict, optional): PSTH smoothing parameters - passed to get_firing_rates and trial_psth. Defaults to None.
        event_time_groups (list, optional): What grouping of conditions to use for calculating relative event timing. Defaults to None.
        stacked (bool, optional): stack data across sessions into single RatePop. Defaults to True.
        return_averaged (bool, optional): If true, calculate trial-average firing rates for each condition in tr_table.
        If false, return single trial data for all trials matching a condition in tr_table. Defaults to True.

    Returns:
        list, or single Rate Pop: if stacked, returns a single RatePopulation object, otherwise, returns a list of RatePop objects - one per entry in popns
    """

    num_sessions, num_alignments = len(popns), len(t_params['align_ev'])

    # --------------------------------
    # extract firing rates as unit x conditions x times, from each recording population's spiking activity

    t_params_a = {k: v for k, v in t_params.items() if k in ['align_ev', 'trange', 'binsize']}
    # TODO alternative version which takes already saved tuple of inputs
    fr_list, unitlabels, conds_dfs, tvecs, _ = zip(*popns.apply(
        lambda x: x.get_firing_rates(**t_params_a, sm_params=smooth_params, condlabels=tr_table.columns)
        )
    )

    # --------------------------------
    # average trials within each condition specified in tr_table
    if return_averaged:
        if t_params['binsize'] > 0:
            rates_cat, _, len_intervals = concat_aligned_rates(fr_list, tvecs)
        else:
            rates_cat, _, len_intervals = concat_aligned_rates(fr_list)

        cond_frs, cond_groups = [], []
        for f_in, cond in zip(rates_cat, conds_dfs):

            # avg firing rate over time/alignments across units, for each cond
            f_out, _, cg = condition_averages(f_in, cond, cond_groups=tr_table)
            cond_frs.append(f_out)
            cond_groups.append(cg)

        # resplit by len_intervals, for pseudopop creation (and overwrite fr_list from individual trials)
        if t_params['binsize'] > 0:
            fr_list = list(map(lambda f, x: np.split(f, x, axis=2)[:-1], cond_frs, len_intervals))
        else:
            fr_list = list(map(lambda f: np.split(f, f.shape[2], axis=2), cond_frs))

        conds_dfs = cond_groups

    # --------------------------------
    # calculate median timing of "other events" relative to alignment event, for each session

    rel_event_times = [None for _ in popns]
    if 'other_ev' in t_params:

        cg_events = None
        if return_averaged:
            if event_time_groups:
                cg_events = tr_table[event_time_groups].drop_duplicates()
            else:
                cg_events = tr_table

        rel_event_times = popns.apply(
            lambda x: x.popn_rel_event_times(align=t_params['align_ev'],
                                             others=t_params['other_ev'],
                                             cond_groups=cg_events
                                            )
                                         )
        rel_event_times = list(rel_event_times)

    # conds_dfs = [df.assign(trialNum=np.arange(len(df))) for df in conds_dfs]

    # --------------------------------
    # build the dataset

    if not stacked:

        pseudo_pop = []
        for i, frs in enumerate(fr_list):

            rate_pop = RatePop(
                subject=popns[0].subject,
                rates_averaged=return_averaged,
                simul_recorded=True,
                firing_rates=frs,
                timestamps=tvecs[i],
                psth_params=t_params,
                rel_events=rel_event_times[i],
                conds=conds_dfs[i],
                clus_group=unitlabels[i],
                area=popns.iloc[i].area,
            )

            pseudo_pop.append(rate_pop)

    else:

        if t_params['binsize'] == 0:
            stacked_frs, t_unq, num_units = stack_firing_rates(fr_list)
        else:
            stacked_frs, t_unq, num_units = stack_firing_rates(fr_list, tvecs)


        # list of area, session number, and unit number within session, for each unit
        area = np.array([p.area for n, p in zip(num_units, popns) for _ in range(n)])
        u_idx = np.array([i for i, n in enumerate(num_units) for _ in range(n)])

        # make conditions list the same size as units (replicate conditions list for units within the same session)
        # stacked_conds = [conds_dfs[u] for u in u_idx]

        # for trial-averaged data, cond_groups is the same for each session
        if return_averaged:
            conds_dfs = conds_dfs[0]

        pseudo_pop = RatePop(
            subject=popns[0].subject,
            rates_averaged=return_averaged,
            simul_recorded=False,
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


def stack_rate_populations(rate_populations: list) -> RatePop:
    """
    takes a list containing neural populations (RatePop objects) from different sessions,
    and concatenates them to form a single RatePop object, i.e. a pseudo-population

    Args:
        rate_populations (list): list of RatePop objects

    Returns:
        RatePop: single RatePop object, "pseudopopulation" of all RatePops in rate_populations
    """

    rel_events = [x.rel_events for x in rate_populations]
    conds = [x.conds for x in rate_populations]
    clus_group = np.hstack([x.clus_group for x in rate_populations])

    if rate_populations[0].rates_averaged:
        conds = conds[0]

    firing_rates = [x.firing_rates for x in rate_populations]
    tvecs = [x.timestamps for x in rate_populations]

    if rate_populations[0].psth_params['binsize'] == 0:
        firing_rates, _, num_units = stack_firing_rates(firing_rates)
    else:
        firing_rates, tvecs, num_units = stack_firing_rates(firing_rates, tvecs)

    area = np.array([p.area for n, p in zip(num_units, rate_populations) for _ in range(n)])
    u_idx = np.array([i for i, n in enumerate(num_units) for _ in range(n)])


    stacked_pop = RatePop(
        subject=rate_populations[0].subject,
        rates_averaged=rate_populations[0].rates_averaged,
        simul_recorded=rate_populations[0].simul_recorded,
        firing_rates=firing_rates,
        timestamps=tvecs,
        psth_params=rate_populations[0].psth_params,
        rel_events=rel_events,
        conds=conds,
        clus_group=clus_group,
        area=area,
        unit_session=u_idx,
    )

    return stacked_pop


def stack_firing_rates(fr_list: Sequence, tvecs: Optional[Sequence] = None):
    """
    stacks firing rates from individual sessions into single array
    will maintain separate lists for individual alignments

    Args:
        fr_list (Sequence): _description_
        tvecs (Optional[Sequence], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # define some useful variables for later on
    num_sessions, num_alignments = len(fr_list), len(fr_list[0])

    num_units = np.array([x[0].shape[0] for x in fr_list]) # number of units per session
    max_trs = max(list(map(lambda x: x[0].shape[1], fr_list))) # max number of trials (or conditions) across all sessions

    stacked_frs, t_unq, t_idx = [], [], []

    # to stack all frs with time-resolutions preserved, make a single unique time vector (t_unq)
    # and insert each population fr matrix according to how its specific tvec lines up with t_unq
    # this is necessary to handle variable start/end references and make sure sessions are lined up correctly
    # - different sessions might have slightly different lengths
    # e.g. motionOn - motionOff varies on each trial
    # (only do it if tvecs is specified, otherwise assume use of interval averages)

    # NOTE unstacked firing rates will come in as a nested lists of lists, with outer list elements for each session
    # and inner list elements for alignments to each individual task event (could just be 1)
    # stacked firing rates will essentially stack over the outer list, but maintain the alignments list

    # loop over alignments, this becomes the top level (only) list
    for j in range(num_alignments):

        u_pos = 0
        if tvecs is not None:
            # time-resolution provided, need to consider temporal alignment
            concat_tvecs = [tvecs[i][j] for i in range(num_sessions)]
            t_unq.append(np.unique(np.concatenate(concat_tvecs)))
            t_idx.append([np.ravel(np.where(np.isin(t, t_unq[j]))) for t in concat_tvecs])

            stacked_frs.append(np.full([num_units.sum(), max_trs, len(t_unq[j])], np.nan))
            for sess in range(num_sessions):
                sess_align_frs = fr_list[sess][j] # firing rates for this session and alignment
                stacked_frs[j][u_pos:u_pos+num_units[sess], 0:sess_align_frs.shape[1], t_idx[j][sess]] = sess_align_frs
                u_pos += num_units[sess]

        else:
            # no time-resolution (just total firing rates in intervals), this is a bit simpler
            stacked_frs.append(np.full([num_units.sum(), max_trs], np.nan))
            for sess in range(num_sessions):
                sess_align_frs = fr_list[sess][j] # firing rates for this session and alignment
                stacked_frs[j][u_pos:u_pos+num_units[sess], 0:sess_align_frs.shape[1]] = np.squeeze(sess_align_frs)
                u_pos += num_units[sess]

    return stacked_frs, t_unq, num_units


# %% ----------------------------------------------------------------

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


# %% ----------------------------------------------------------------
# convert cluster group int into cluster label

def get_cluster_label(clus_group, labels=['unsorted', 'mua', 'good', 'noise']):

    try:
        return labels.index(clus_group)
    except ValueError:
        return None


# %% ----------------------------------------------------------------
# build 'raw' dataset

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

    pars = ['Tuning', 'Task', 'RFMapping', 'VesMapping']

    rec_df = rec_info.copy(deep=True)
    rec_df[pars] = pd.NA

    print(f"Creating neural dataset for {subject}, paradigms: {par_labels}, n = {len(data)}\n")
    for index, sess in enumerate(data):

        rec_date = sess['date']
        rec_set = sess['rec_set']
        rec_folder = PurePath(data_folder, rec_date)  # just for bookkeeping

        rec_sess_info = rec_info.iloc[index, :].to_dict()
        rec_sess_info['chs'] = np.arange(rec_sess_info['min_ch'], rec_sess_info['max_ch']+1)
        rec_sess_info['grid_xy'] = (rec_sess_info['grid_x'], rec_sess_info['grid_y'])

        if not rec_sess_info['is_good']:
            continue

        print(f"Adding {rec_date}, set {rec_set}, {rec_sess_info['area']}\n")

        for p, par in enumerate(pars):

            if isinstance(sess['data'], dict) and par_labels[p] in sess['data'].keys():

                rec_popn = build_rec_popn(
                    subject, rec_date, rec_sess_info, data=sess['data'][par_labels[p]],
                    )

                rec_df.loc[index, par] = rec_popn

    if save_file:
        filename = f"{Path(data_file).stem}.pkl"
        rec_df.to_pickle(filename)

    return rec_df




if __name__ == '__main__':

    mat_data_file = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/data/lucio_neuro_datasets/lucio_20220512-20230602_neuralData.mat'
    rec_info_file = '/Users/stevenjerjian/Desktop/FetschLab/Analysis/info/RecSessionInfo.xlsx'

    create_dataset(mat_data_file, rec_info_file)



