import numpy as np
import pandas as pd

def dots3DMP_create_trial_list(hdgs, mods: list = [1, 2, 3],
                               cohs: list = [1], deltas=0,
                               nreps: int = 1, shuff: bool = True):

    # if shuff:
    #     np.random.seed(42)  # for reproducibility

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
    condlist = np.column_stack((modality, coh, hdg, delta))
    trial_table = np.tile(condlist, (nreps, 1))
    ntrials = len(trial_table)

    if shuff:
        trial_table = trial_table[np.random.permutation(ntrials)]

    trial_table = pd.DataFrame(trial_table, columns=['modality', 'coherence', 'heading', 'delta'])

    return trial_table, ntrials





def main():
    mods = np.array([1, 2, 3])
    cohs = np.array([1, 2])
    hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
    deltas = np.array([-3, 0, 3])
    nreps = 1

    trial_table, ntrials = \
        dots3DMP_create_trial_list(hdgs, mods, cohs, deltas,
                                   nreps, shuff=False)

    return trial_table, ntrials

if __name__ == '__main__':
    main()