#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:00:07 2023

@author: stevenjerjian
"""

from ddm_moi import ddm_2d
from ddm_moi.Accumulator import AccumulatorModelMOI, log_pmap
import numpy as np
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(nrows=2, ncols=2)

# ax[0, 0].plot(accumulator.drift_labels, accumulator.p_corr)
# ax[0, 0].set_xlabel('headings')
# ax[1, 0].plot(orig_tvec, accumulator.rt_dist.T)
# ax[1, 0].set_xlabel('time')
# ax[1, 1].plot(accumulator.tvec, accumulator.rt_dist.T)
# plt.show()

# plt.plot(np.linspace(0, 2, 400), full_dvs[:, 0, (sim_data['modality']==2) & (sim_data['heading']==12)], color='r')
# plt.plot(np.linspace(0, 2, 400), full_dvs[:, 0, (sim_data['modality']==1) & (sim_data['heading']==12)], color='k')


# %%

params = {
    'kmult': [0.08, 0.3],
    'bound': 0.8,
    'theta': 1.1,
}

accum = {'tvec': np.arange(0, 2, 0.01),  # 0.005
         'grid_vec': np.arange(-3, 0, 0.025),  # 0.025
        }

hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
# hdgs = np.array([3])

mod_labels = ['ves', 'vis', 'comb']
mod = 1
delta = 0

stim_sc = ddm_2d.get_stim_urgs(tvec=accum['tvec'])
accum_result, orig_tvec = ddm_2d.dots3dmp_accumulator(params, hdgs, mod, delta, accum_kw=accum, stim_scaling=stim_sc, use_signed_drifts=False)
accum_result.plot()
# accum_result.pdf(full_pdf=True)

# %%

h = 1

N = 10
fig, axs = plt.subplots(1, N, figsize=(14, 2), sharey=True)

time_pts = np.round(np.linspace(1, 150, N)).astype(int)
for t, tpt in enumerate(time_pts):
    z = np.squeeze(accum_result.pdf3D[h, tpt, :, :])
    cont = axs[t].contourf(accum_result.grid_vec, accum_result.grid_vec, z, levels=100)
    axs[t].set_title(f'{orig_tvec[tpt]}s')
    axs[t].set_aspect('equal')
plt.colorbar(cont, ax=axs)
# plt.tight_layout()

fig.suptitle(f"{mod_labels[mod-1]}, heading = {hdgs[h]}")
plt.show()

# %% Test effect of params on RT distribution

params = {
    'kmult': [0.3, 0.3],
    'bound': 0.75,
    'theta': 1.1,
    'ndt': 0.2,
    'sigma_ndt': 0.05,
}

accum = {'tvec': np.arange(0, 2, 0.01),  # 0.005
        'grid_vec': np.arange(-3, 0, 0.025),  # 0.025
        }

# hdgs = np.array([-12, -6, -3, -1.5, 0, 1.5, 3, 6, 12])
hdgs = np.array([3])
mod_labels = ['ves','vis','comb']
mods = [1, 2, 3]

delta = 0

stim_sc = ddm_2d.get_stim_urgs(tvec=accum['tvec'])



# %% kmults spread for diff bounds

kmults = np.arange(0.1, 2.6, 0.4)
kmults = kmults[::-1]  # for better plotting
params['ndt'] = 0.3

kmult_labels = [f'{k:.2f}' for k in kmults]
bounds = [0.3, 0.6, 0.9, 1.2]

fig1, ax = plt.subplots(3, len(bounds), sharex=True, sharey=True, figsize=(10, 10))
rt_means = np.zeros((len(mods), len(bounds), len(kmults)))

for m, mod in enumerate(mods):
    for b, bound in enumerate(bounds):
        params['bound'] = bound
        rt_dists = []
        for k, km in enumerate(kmults):
            params['kmult'] = [km, km]
            accum_result, orig_tvec = ddm_2d.dots3dmp_accumulator(params, hdgs, mod, delta, accum_kw=accum, stim_scaling=stim_sc, use_signed_drifts=False)
            rt_dists.append(accum_result.rt_dist)
            rt_means[m, b, k] = np.sum(accum_result.rt_dist * orig_tvec)

        rt_dists = np.vstack(rt_dists)

        ax[m][b].plot(orig_tvec, rt_dists.T)

        if b == 0:
            ax[m][b].set_ylabel(mod_labels[m])
            if m == 0:
                ax[m][b].legend(kmult_labels, frameon=False, fontsize=8, loc='right', labelcolor='linecolor')

        elif b == 1:
            if m == 2:
                ax[m][b].set_xlabel('time [s]')
        if m == 0:
            ax[m][b].set_title(f"bound={bound}")

        ax[m][b].plot(rt_means[m, b, :], [0.1]*len(kmults), marker='.', markersize=5, color='k')


fig1.suptitle('RT distribution for different kmults')
plt.show()


# %% bounds spread for diff kmults

kmults = [0.1, 0.5, 1.0, 1.5]
bounds = np.arange(0.2, 1.3, 0.2)
params['ndt'] = 0.3

bound_labels = [f'{b:.2f}' for b in bounds]

fig2, ax = plt.subplots(3, len(kmults), sharex=True, sharey=True, figsize=(10, 10))
rt_means = np.zeros((len(mods), len(bounds), len(kmults)))

for m, mod in enumerate(mods):
    for k, km in enumerate(kmults):
        params['kmult'][0] = km
        rt_dists = []
        for b, bound in enumerate(bounds):
            params['bound'] = bound
            accum_result, orig_tvec = ddm_2d.dots3dmp_accumulator(params, hdgs, mod, delta, accum_kw=accum, stim_scaling=stim_sc, use_signed_drifts=False)
            rt_dists.append(accum_result.rt_dist)
            rt_means[m, b, k] = np.sum(accum_result.rt_dist * orig_tvec)

        rt_dists = np.vstack(rt_dists)

        ax[m][k].plot(orig_tvec, rt_dists.T)

        if k == 0:
            ax[m][k].set_ylabel(mod_labels[m])
            if m == 0:
                ax[m][k].legend(bound_labels, frameon=False, fontsize=8, loc='center left', labelcolor='linecolor')
        elif k == 1:
            if m == 2:
                ax[m][k].set_xlabel('time [s]')
        if m == 0:
            ax[m][k].set_title(f"kmult={km:.2f}")

        ax[m][k].plot(rt_means[m, :, k], [0.1]*len(bounds), marker='.', markersize=5, color='k')


fig2.suptitle('RT distribution for different bounds')
plt.show()

# %% different subplots for bounds, same for ndt

ndts = np.linspace(0.1, 0.5, 5)
bounds = [0.3, 0.6, 0.9, 1.2]
params['kmult'] = [0.5, 0.5]

ndt_labels = [f'{n:.2f}' for n in ndts]

fig3, ax = plt.subplots(3, len(bounds), sharex=True, sharey=True,  figsize=(10, 10))

for m, mod in enumerate(mods):
    for b, bound in enumerate(bounds):
        params['bound'] = bound
        rt_dists = []
        for n in ndts:
            params['ndt'] = n
            accum_result, orig_tvec = ddm_2d.dots3dmp_accumulator(params, hdgs, mod, delta, accum_kw=accum, stim_scaling=stim_sc, use_signed_drifts=False)
            rt_dists.append(accum_result.rt_dist)
        rt_dists = np.vstack(rt_dists)

        ax[m][b].plot(orig_tvec, rt_dists.T)

        if b == 0:
            ax[m][b].set_ylabel(mod_labels[m])
            if m == 0:
                ax[m][b].legend(ndt_labels, frameon=False, fontsize=8, loc='center right', labelcolor='linecolor')

        elif b == 1:
            if m == 2:
                ax[m][b].set_xlabel('time [s]')
        if m == 0:
            ax[m][b].set_title(f"bound={bound:.2f}")



fig3.suptitle(f"RT distribution for different non-decision times, kmult={params['kmult'][0]:.1f}")
plt.show()

# %%
# # sensitivity time-course
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(accumulator.tvec, b_ves, color='k', label='ves')
# axs[0].plot(accumulator.tvec, b_vis, color='r', label='vis')
# axs[0].set_title('Stimulus profile')
# axs[0].legend()

# axs[1].plot(accumulator.tvec, cumul_bves, color='k', label='ves')
# axs[1].plot(accumulator.tvec, cumul_bvis, color='r', label='vis')
# axs[1].set_title('Cumulative sensitivity time-course')

# plt.xlabel('time [s]')

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].hist(actual_rts, bins=20)
# ax[1].hist(samp_rts)
# ax[2].plot(orig_tvec, rt_dist)
