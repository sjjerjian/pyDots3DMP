
import numpy as np
import matplotlib.pyplot as plt

from ddm_moi.Accumulator import AccumulatorModelMOI
from ddm_moi.ddm_2d import get_stim_urg

accum = AccumulatorModelMOI(
    tvec=np.arange(0, 2, 0.05),
    grid_vec=np.arange(-3, 0, 0.025),
    sensitivity=30*0.2,
)

hdgs = np.array(np.linspace(-12, 12, 9))  # to plot a smooth fit
hdgs = np.sin(np.deg2rad(hdgs))


accum.bound = np.array([2, 2])
accum.urgency = get_stim_urg(tvec=accum.tvec, moment=1)
accum.set_drifts(list(hdgs))
accum.dist()
print(accum.p_corr)

plt.plot(accum.drift_rates[0])
plt.show()

plt.plot(accum.tvec, accum.rt_dist.T)
plt.show()

accum.bound = np.array([2, 2])
accum.urgency = None
accum.set_drifts(list(hdgs))
accum.dist()
print(accum.p_corr)

plt.plot(accum.tvec, accum.rt_dist.T)
plt.show()
