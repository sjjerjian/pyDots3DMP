import numpy as np
from scipy.stats import multivariate_normal as mvn
from statsmodels.sandbox.distributions.extras import mvnormcdf as mvnormcdf

import matplotlib.pyplot as plt
from matplotlib import animation

from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Optional
from codetiming import Timer

import time

import threading


def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)

# https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays
def np_cache(function):
    @lru_cache
    def cached_wrapper(*args, **kwargs):

        args = [np.array(a) if isinstance(a, tuple) and not isinstance(a, (int, float)) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) and not isinstance(v, (int, float)) else v for k, v in kwargs.items()
        }

        # call the function now, with the array arguments
        return function(*args, **kwargs)

    # wrapper to convert array inputs to hashables, for caching
    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(tuple(row) for row in a) if isinstance(a, np.ndarray) and a.ndim > 1 else tuple(a) for a in args]
        kwargs = {
            k: tuple(tuple(row) for row in v) if isinstance(v, np.ndarray) and v.ndim > 1 else tuple(v) for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def np_cache_minimal(function):
    @lru_cache
    def cached_wrapper(*args, **kwargs):

        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        # call the function now, with the array arguments
        return function(*args, **kwargs)

    # wrapper to convert array inputs to hashables, for caching
    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@dataclass(repr=False)
class AccumulatorModelMOI:
    """
    Dataclass for accumulator model calculated via method of images
    Initialized with certain parameters, then can calculate pdf, logoddsmap, cdf etc, using class methods

    # TODO clean up documentation
    # TODO allow to get cdf or pdf

    """

    bound: np.ndarray = np.array([1, 1])
    tvec: np.ndarray = field(default=np.arange(0, 2, 0.005))

    # don't initialize, they will get set by the setter methods!
    _bound: np.ndarray = field(init=False, repr=False)
    _tvec: np.ndarray = field(init=False, repr=False)

    dt: float = field(init=False, repr=False)

    grid_spacing: float = field(default=0.025)
    drift_rates: list = field(default_factory=list)
    drift_labels: list = field(default_factory=list)
    sensitivity: float = field(default=1)
    urgency: np.ndarray = field(default=None)
    num_images: int = 7

    # TODO clean this up a bit, if we can?
    grid_vec: np.ndarray = np.array([])
    p_corr: np.ndarray = np.array([])
    rt_dist: np.ndarray = np.array([])
    pdf3D: np.ndarray = np.array([])
    up_lose_pdf: np.ndarray = np.array([])
    lo_lose_pdf: np.ndarray = np.array([])
    log_odds: np.ndarray = np.array([])



    @property
    def bound(self):
        return self._bound

    @bound.setter
    def bound(self, b):
        if isinstance(b, (int, float)):
            b = [b, b]
        self._bound = np.array(b)

    @property
    def tvec(self):
        return self._tvec

    @tvec.setter
    def tvec(self, time_vec: np.ndarray):
        self._tvec = time_vec
        self.dt = np.gradient(time_vec)


    def set_drifts(self, drifts: Optional[list] = None, labels: Optional[list] = None):

        if drifts is not None:
            self.drift_rates = drifts

        # add corresponding negated value for anti-correlated accumulator
        # also update drift rates based on sensitivity and urgency, if provided
        for d, drift in enumerate(self.drift_rates):
            drift = drift * np.array([1, -1])
            self.drift_rates[d] = urgency_scaling(drift * self.sensitivity, self.tvec, self.urgency)


        if labels is not None:
            self.drift_labels = labels

        return self


    def __post_init__(self):

        # default set the drift labels as 0:ndrifts
        if len(self.drift_labels) == 0:
             self.drift_labels = np.arange(len(self.drift_rates))
        self.set_drifts(labels=self.drift_labels)


    def _pdf(self, full_pdf=False):

        # TODO allow flexible specification of grid_vec, to use mgrid
        if full_pdf:
            xmesh, ymesh = np.meshgrid(self.grid_vec, self.grid_vec)
        else:
            xmesh1, ymesh1 = np.meshgrid(self.grid_vec, self.grid_vec[-1])
            xmesh2, ymesh2 = np.meshgrid(self.grid_vec[-1], self.grid_vec)

        pdfs, marg_up, marg_lo = [], [], []

        for drift in self.drift_rates:

            if full_pdf:
                pdf_3d = moi_pdf(xmesh, ymesh, self.tvec, drift, self.bound, self.num_images)
                pdfs.append(pdf_3d)

                pdf_up = pdf_3d[:, :, -1]  # right bound
                pdf_lo = pdf_3d[:, -1, :]  # top bound

            else:
                # only need to calculate pdf at the boundaries!
                pdf_lo = moi_pdf(xmesh1, ymesh1, self.tvec, drift, self.bound, self.num_images)
                pdf_up = moi_pdf(xmesh2, ymesh2, self.tvec, drift, self.bound, self.num_images)

            # distribution of losing accumulator, GIVEN that winner has hit bound
            marg_up.append(pdf_up)  # right bound
            marg_lo.append(pdf_lo)  # top bound

        if full_pdf:
            self.pdf3D = np.stack(pdfs, axis=0)

        self.up_lose_pdf = np.stack(marg_up, axis=0)
        self.lo_lose_pdf = np.stack(marg_lo, axis=0)

        return self


    def log_posterior_odds(self):
        self.log_odds = log_odds(self.up_lose_pdf, self.lo_lose_pdf)
        return self


    def _cdf(self):
        p_corr, rt_dist = [], []
        for drift in self.drift_rates:
            p_up, rt, flux1, flux2 = moi_cdf(self.tvec, drift, self.bound, 0.025, self.num_images)
            p_corr.append(p_up)
            rt_dist.append(rt)

        self.p_corr = np.array(p_corr)
        self.rt_dist = np.stack(rt_dist, axis=0)

        return self


    def dv(self, drift, sigma, cusum=False):
        return moi_dv(mu=drift*self.tvec.reshape(-1, 1), s=sigma, num_images=self.num_images)


    def dist(self, return_pdf=False):
        self._cdf()

        if return_pdf:
            self._pdf()

        return self


    def plot(self, d_ind=-1):

        # d_ind - index of which drift to plot for PDFs

        fig_cdf, axc = plt.subplots(2, 1, figsize=(4, 5))
        axc[0].plot(self.drift_labels, self.p_corr)
        axc[0].set_xlabel('drift')
        axc[0].set_xticks(self.drift_labels)
        axc[0].set_ylabel('prob. correct choice')

        axc[1].plot(self.tvec, self.rt_dist.T)
        axc[1].legend(self.drift_labels)
        axc[1].set_xlabel('Time (s)')
        axc[1].set_title('RT distribution (no NDT)')
        fig_cdf.tight_layout()

        fig_pdf = None
        if self.up_lose_pdf:
            fig_pdf, axp = plt.subplots(2+self.log_odds, 1, figsize=(5, 6))
            contour = axp[0].contourf(self.tvec, self.grid_vec,
                                      log_pmap(np.squeeze(self.up_lose_pdf[d_ind, :, :])).T,
                                      levels=100)
            axp[1].contourf(self.tvec, self.grid_vec,
                            log_pmap(np.squeeze(self.lo_lose_pdf[d_ind, :, :])).T,
                            levels=100)
            axp[0].set_title(f"Losing accumulator | Correct, drift rate {self.drift_labels[d_ind]}")
            axp[1].set_title(f"Losing accumulator | Error, drift rate {self.drift_labels[d_ind]}")
            cbar = fig_cdf.colorbar(contour, ax=axp[0])
            cbar = fig_cdf.colorbar(contour, ax=axp[1])

            if self.log_odds:
                vmin, vmax = 0, 3
                contour = axp[2].contourf(self.tvec, self.grid_vec,
                                          self.log_odds.T, vmin=vmin, vmax=vmax,
                                          levels=100)
                axp[2].set_title("Log Odds of Correct Choice given Losing Accumulator")
                cbar = fig_pdf.colorbar(contour, ax=axp[2])
            fig_pdf.tight_layout()

        return fig_cdf, fig_pdf


    def plot_3d(self, d_ind=-1):

        # fig, ax = plt.subplots()
        # cont = plt.contourf(self.grid_vec, self.grid_vec, self.pdf3D[d_ind, 0, :, :])
        # ax.set_title(f't = {self.tvec[i]:.2f}')
        # fig.canvas.draw()

        # for i in range(1, len(self.tvec)):
        #     cont = plt.contourf(self.grid_vec, self.grid_vec, self.pdf3D[d_ind, i, :, :])
        #     ax.set_title(f't = {self.tvec[i]:.2f}')
        #     fig.canvas.draw()

        # plt.show()


        def animate_wrap(i):
            z = self.pdf3D[d_ind, i, :, :]
            cont = plt.contourf(self.grid_vec, self.grid_vec, z)
            return cont

        def init():
            cont = plt.contourf(self.grid_vec, self.grid_vec, self.pdf3D[d_ind, 0, :, :])
            return cont

        anim_event = threading.Event()

        fig, ax = plt.subplots()
        anim = animation.FuncAnimation(fig, animate_wrap, frames=len(self.tvec),
                                       init_func=init, interval=50, repeat=False)

        anim_event.wait()

        # TODO video/gif style plot of third quadrant pdf at each timestep




# ============
# functions

@np_cache_minimal
def _sj_rot(j, s0, k):
    """

    :param j: jth image
    :param s0: starting_point, length 2 array
    :param k: 2*k-1 is the number of images
    :return:
    """

    alpha = (k - 1)/k * np.pi
    sin_alpha = np.sin(j * alpha)
    sin_alpha_plus_k = np.sin(j * alpha + np.pi / k)
    sin_alpha_minus_k = np.sin(j * alpha - np.pi / k)

    if j % 2 == 0:
        s = np.array([[sin_alpha_plus_k, sin_alpha], [-sin_alpha, -sin_alpha_minus_k]])
    else:
        s = np.array([[sin_alpha, sin_alpha_minus_k], [-sin_alpha_plus_k, -sin_alpha]])

    return (1 / np.sin(np.pi / k)) * (s @ s0.T)


def _weightj(j, mu, sigma, sj, s0):
    """

    :param j: jth image
    :param mu: drift rate
    :param sigma: covariance
    :param sj: output from sj_rot()
    :param s0: starting_point, length 2 array
    :return: image weight
    """

    return (-1) ** j * np.exp(mu @ np.linalg.inv(sigma) @ (sj - s0).T)


@lru_cache(maxsize=32)
def _corr_num_images(num_images):
    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    return sigma, k


def moi_pdf(xmesh: np.ndarray, ymesh: np.ndarray, tvec: np.ndarray,
            mu: np.ndarray, bound=np.array([1, 1]), num_images: int=7):
    """

    :param xmesh:
    :param ymesh:
    :param tvec: 1-D array containing times to evaluate pdf
    :param mu: drift rate 2xlen(tvec) array (to incorporate any urgency signal)
    :param bound: bound, length 2 array
    :param num_images:
    :return:
    """
    sigma, k = _corr_num_images(num_images)

    nx, ny = xmesh.shape
    pdf_result = np.zeros((len(tvec), nx, ny)).squeeze()

    xy_mesh = np.dstack((xmesh, ymesh))

    s0 = -bound
    # skip the first sample (t starts at 1)
    for t in range(1, len(tvec)):

        pdf_result[t, ...] = pdf_at_timestep(tvec[t], mu[t, :], sigma, xy_mesh, k, s0)

        # pdf_result[t, ...] = mvn(mean=s0 + mu_t, cov=sigma_t).pdf(xy_mesh)
        # for j in range(1, k*2):
        #     sj = sj_rot(j, s0, k)
        #     a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
        #     pdf_result[t, ...] += a_j * mvn(mean=sj + mu_t, cov=sigma_t).pdf(xy_mesh)

    return pdf_result


@np_cache
def _mvn_timestep(mean: np.ndarray, cov: np.ndarray):
    return mvn(mean=mean, cov=cov)


#@np_cache
def pdf_at_timestep(t, mu, sigma, xy_mesh, k, s0):

    # pdf = mvn(mean=s0 + mu*t, cov=sigma*t).pdf(xy_mesh)
    pdf = _mvn_timestep(mean=s0 + mu*t, cov=sigma*t).pdf(xy_mesh)

    for j in range(1, k * 2):
        sj = _sj_rot(j, s0, k)
        a_j = _weightj(j, mu.T, sigma, sj, s0)
        # pdf += a_j * mvn(mean=sj + mu*t, cov=sigma*t).pdf(xy_mesh)
        pdf += a_j * _mvn_timestep(mean=sj + mu*t, cov=sigma*t).pdf(xy_mesh)

    return pdf


#@Timer(name='moi_cdf')
def moi_cdf(tvec: np.ndarray, mu, bound=np.array([1, 1]), margin_width=0.025, num_images: int = 7):
    """
    For a given 2-D particle accumulator with drift mu over time tvec, calculate
        a) the probability of a correct choice
        b) the distribution of response times (bound crossings)
    choices are calculated by evaluating cdf at each boundary separately,
    rt_dist is calculated agnostic to choice.
    :param tvec: 1-D array containing times to evaluate pdf
    :param mu: drift rate 2xlen(tvec) array (to incorporate any urgency signal)
    :param bound: default [1 1]
    :param num_images: number of images for method of images, default 7
    :return: probability of correct choice (p_up), and decision time distribution (rt_dist)

    TODO see how much changing the difference between bound and bound_marginal affects anything

    """

    sigma, k = _corr_num_images(num_images)

    survival_prob = np.ones(len(tvec))
    flux1, flux2 = np.zeros(len(tvec)), np.zeros(len(tvec))

    s0 = -bound
    b0, bm = -margin_width, 0
    bound0 = np.array([b0, b0])
    bound1 = np.array([b0, bm])  # top boundary of third quadrant
    bound2 = np.array([bm, b0])  # right boundary

    # skip the first sample (t starts at 1)
    for t in range(1, len(tvec)):

        # why do we need this?
        if tvec[t] == 0:
            # tvec[t] = np.finfo(np.float64).eps
            tvec[t] = np.min(tvec[tvec > 0])

        mu_t = mu[t, :].T * tvec[t]

        # total density within boundaries

        mvn_0 = _mvn_timestep(mean=s0 + mu_t, cov=sigma * tvec[t])
        mvn_0.maxpts = 10000*2

        cdf_rest = mvn_0.cdf(bound0)

        # density beyond boundary, in one or other direction
        cdf1 = mvn_0.cdf(bound1) - cdf_rest
        cdf2 = mvn_0.cdf(bound2) - cdf_rest

        # loop over images
        for j in range(1, k*2):
            sj = _sj_rot(j, s0, k)

            mvn_j = _mvn_timestep(mean=sj + mu_t, cov=sigma * tvec[t])
            mvn_j.maxpts = 10000*2


            # total density WITHIN boundaries for jth image
            cdf_add = mvn_j.cdf(bound0)

            # density BEYOND boundary in one or other direction, for jth image
            cdf_add1 = mvn_j.cdf(bound1) - cdf_add
            cdf_add2 = mvn_j.cdf(bound2) - cdf_add

            a_j = _weightj(j, mu[t, :].T, sigma, sj, s0)
            cdf_rest += (a_j * cdf_add)
            cdf1 += (a_j * cdf_add1)
            cdf2 += (a_j * cdf_add2)

        survival_prob[t] = cdf_rest
        flux1[t] = cdf1
        flux2[t] = cdf2

    p_up = np.sum(flux2) / np.sum(flux1 + flux2)

    # if we want the correct vs error RT distributions,
    # then presumably can treat flux1 and flux2 as 1-survival_probs
    rt_dist = np.diff(np.insert(1-survival_prob, 0, 0))

    # winning and losing pdfs? kinda
    # pdf_up = np.diff(flux2)
    # pdf_lo = np.diff(flux1)

    return p_up, rt_dist, flux1, flux2


def moi_dv(mu: np.ndarray, s: np.ndarray = np.array([1, 1]), num_images: int = 7) -> np.ndarray:

    sigma, k = _corr_num_images(num_images)

    V = np.diag(s) * sigma * np.diag(s)

    dv = np.zeros_like(mu)

    # FIXME for some reason this seems to produce the same sequences each run
    # would be good to control this more explicitly with seed setting
    for t in range(1, mu.shape[0]):
        dv[t, :] = mvn(mu[t, :].T, cov=V).rvs()

    dv = dv.cumsum(axis=0)

    return dv


def urgency_scaling(mu: np.ndarray, tvec: np.ndarray, urg=None) -> np.ndarray:

    if len(mu) != len(tvec):
        mu = np.tile(mu, (len(tvec), 1))

    if urg is not None:
        if isinstance(urg, (int, float)):
            urg = np.ones(len(tvec)-1) * urg/(len(tvec)-1)
            urg = np.insert(urg, 0, 0)

        assert len(urg) == len(tvec) == len(mu),\
            "If urgency signal is a vector, it must match tvec and drift vector lengths"

        mu = mu + urg.reshape(-1, 1)

    return mu


def log_odds(pdf1: np.ndarray, pdf2: np.ndarray) -> np.ndarray:
    """
    Calculate log posterior odds of correct choice.

    assumes that drift is the first dimension, which gets marginalized over
    :param pdf1: pdf of losing race for correct trials
    :param pdf2: pdf of losing race for incorrect trials
    :return log_odds_correct: heatmap, log posterior odds of correct choice
    """
    # replaces zeros with tiny value to avoid logarithm issues
    min_val = np.finfo(np.float64).tiny
    pdf1 = np.clip(pdf1, a_min=min_val, a_max=None)
    pdf2 = np.clip(pdf2, a_min=min_val, a_max=None)

    odds = np.sum(pdf1, axis=0) / np.sum(pdf2, axis=0)
    odds = np.clip(odds, a_min=1, a_max=None)
    return np.log(odds)


def log_pmap(pdf: np.ndarray, q: int = 30) -> np.ndarray:
    """
    for visualization of losing accumulator pdf, it's likely helpful to perform a cutoff and look at log space
    :param pdf:
    :param q:
    :return:
    """

    pdf = np.clip(pdf, a_min=10**(-q), a_max=None)
    return (np.log10(pdf)+q) / q





