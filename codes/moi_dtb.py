import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from numba import njit, prange
import numba as nb
import time
from dataclasses import dataclass, field

@dataclass
class AccumulatorModelMOI:

    tvec: np.ndarray = field(repr=False)
    grid_vec: np.ndarray
    drift_rates: np.ndarray
    urgency: np.ndarray = field(default=0)
    bound: np.ndarray = field(default=1)
    num_images: int = 7

    grid_xmesh: np.ndarray = field(init=False)
    grid_ymesh: np.ndarray = field(init=False)

    p_corr: np.ndarray = field(init=False)
    rt_dist: np.ndarray = field(init=False)

    pdf3D: np.ndarray = field(init=False)
    up_lose_pdf: np.ndarray = field(init=False)
    lo_lose_pdf: np.ndarray = field(init=False)

    log_odds: np.ndarray = field(init=False)

    def __post_init__(self):
        for d, drift in enumerate(self.drift_rates):
            self.drift_rates[d] = urgency_scaling(drift, self.tvec, self.urgency)

    def pdf(self, return_marginals=True, return_mesh=True):

        # TODO allow flexible specification of grid_vec, to use mgrid
        xmesh, ymesh = np.meshgrid(self.grid_vec, self.grid_vec)

        if return_mesh:
            self.grid_xmesh = xmesh
            self.grid_ymesh = ymesh

        pdfs, marg_up, marg_lo = [], [], []

        for drift in self.drift_rates:

            pdf_3d = moi_pdf(xmesh, ymesh, self.tvec, drift, self.bound, self.num_images)

            if return_marginals is False:
                pdfs.append(pdf_3d)
            else:
                marg_up.append(np.sum(pdf_3d, axis=2))  # sum over y
                marg_lo.append(np.sum(pdf_3d, axis=1))  # sum over x

                # TODO validate this, and return them if we want...
                up_pdf = np.sum(pdf_3d[:, -1, :], axis=1)
                lo_pdf = np.sum(pdf_3d[:, :, -1], axis=1)

                up_cdf = np.cumsum(up_pdf)
                lo_cdf = np.cumsum(lo_pdf)

        if return_marginals is False:
            self.pdf3D = np.stack(pdfs, axis=0)
        else:
            self.up_lose_pdf = np.stack(marg_up, axis=0)
            self.lo_lose_pdf = np.stack(marg_lo, axis=0)

        return self

    def log_posterior_odds(self):
        self.log_odds = log_odds(self.up_lose_pdf, self.lo_lose_pdf)

    def cdf(self):
        p_corr, rt_dist = [], []
        for drift in self.drift_rates:
            p_up, rt = moi_cdf(self.tvec, drift, self.bound, self.num_images)
            p_corr.append(p_up)
            rt_dist.append(rt)

        self.p_corr = np.array(p_corr)
        self.rt_dist = np.stack(rt_dist, axis=0)

        return self

    def fit(self, return_pdf=False):
        self.cdf()

        if return_pdf:
            self.pdf(return_marginals=True).log_posterior_odds()

        return self



    # attrs from methods
    # prob correct
    # rt distribution
    # full 3d pdf
    # losing race pdfs for correct and incorrect
    # log odds map

    # plotting methods


def sj_rot(j, s0, k):
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


def weightj(j, mu, sigma, sj, s0):
    """

    :param j: jth image
    :param mu: drift rate
    :param sigma: covariance
    :param sj: output from sj_rot()
    :param s0: starting_point, length 2 array
    :return: image weight
    """

    return (-1) ** j * np.exp(mu @ np.linalg.inv(sigma) @ (sj - s0).T)

def moi_pdf(xmesh: np.ndarray, ymesh: np.ndarray, tvec: np.ndarray, mu: np.ndarray, bound=np.array([1, 1]), num_images: int=7):
    """

    :param xmesh:
    :param ymesh:
    :param tvec: 1-D array containing times to evaluate pdf
    :param mu: drift rate 2xlen(tvec) array (to incorporate any urgency signal)
    :param bound: bound, length 2 array
    :param num_images:
    :return:
    """
    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    nx, ny = xmesh.shape
    pdf_result = np.empty((len(tvec), nx, ny))

    xy_mesh = np.dstack((xmesh, ymesh))

    s0 = -bound
    # skip the first sample (t starts at 1)
    for t in range(1, len(tvec)):

        # start = time.time()

        mu_t = mu[t, :] * tvec[t]
        sigma_t = sigma * tvec[t]

        pdf_result[t, :, :] = mvn(mean=s0 + mu_t, cov=sigma_t).pdf(xy_mesh)
        for j in range(1, k*2):
            sj = sj_rot(j, s0, k)
            a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
            pdf_result[t, :, :] += a_j * mvn(mean=sj + mu_t, cov=sigma_t).pdf(xy_mesh)

        # end = time.time()
        # if t%30 == 0:
        #    print(f"Time to compute pdf, timestep {t} = {end-start:.4f}")

    return pdf_result


def moi_pdf_decorator(func):
    def wrapper(*args, **kwargs):
        pdf_result = func(*args, **kwargs)


        p_up_lose_pdf = np.squeeze(np.sum(pdf_result, axis=2))  # sum over y
        p_lo_lose_pdf = np.squeeze(np.sum(pdf_result, axis=1))  # sum over x

        p_up_pdf = np.sum(pdf_result[:, -1, :], axis=1)
        p_lo_pdf = np.sum(pdf_result[:, :, -1], axis=1)

        p_up_cdf = np.cumsum(p_up_pdf)
        p_lo_cdf = np.cumsum(p_lo_pdf)

    return wrapper


def moi_cdf(tvec, mu, bound=np.array([1, 1]), num_images: int=7):
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

    TODO see how mcuh changing the difference between bound and bound_marginal affects anything

    """
    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    survival_prob = np.ones(len(tvec))
    flux1, flux2 = np.empty(len(tvec)), np.empty(len(tvec))

    s0 = -bound
    b0, bm = -0.025, 0
    bound0 = np.array([b0, b0])
    bound1 = np.array([b0, bm])  # top boundary of third quadrant
    bound2 = np.array([bm, b0])  # right boundary

    # skip the first sample (t starts at 1)
    for t in range(1, len(tvec)):

        mu_t = mu[t, :].T * tvec[t]

        mvn_0 = mvn(s0 + mu_t, cov=sigma * tvec[t])

        # total density within boundaries
        cdf_rest = mvn_0.cdf(bound0)

        # density beyond boundary in one or other direction
        cdf1 = mvn_0.cdf(bound1) - cdf_rest
        cdf2 = mvn_0.cdf(bound2) - cdf_rest

        # loop over images
        for j in range(1, k*2):
            sj = sj_rot(j, s0, k)

            mvn_j = mvn(sj + mu_t, cov=sigma*tvec[t])

            # total density WITHIN boundaries for jth image
            cdf_add = mvn_j.cdf(bound0)

            # density BEYOND boundary in one or other direction, for jth image
            cdf_add1 = mvn_j.cdf(bound1) - cdf_add
            cdf_add2 = mvn_j.cdf(bound2) - cdf_add

            a_j = weightj(j, mu[t, :].T, sigma, sj, s0)

            cdf_rest += (a_j * cdf_add)
            cdf1 += (a_j * cdf_add1)
            cdf2 += (a_j * cdf_add2)

        survival_prob[t] = cdf_rest
        flux1[t] = cdf1
        flux2[t] = cdf2

    p_up = np.sum(flux2) / np.sum(flux1 + flux2)

    # if we want the correct vs error RT distributions, then presumably can treat flux1 and flux2 as 1-survival_probs
    rt_dist = np.diff(np.insert(1-survival_prob, 0, 0))

    # winning and losing pdfs? kinda
    pdf_up = np.diff(flux2)
    pdf_lo = np.diff(flux1)

    return p_up, rt_dist


def urgency_scaling(mu, tvec, urg):
    """
    :param mu:
    :param tvec:
    :param urg:
    :return:
    """

    # no urgency, just tile mu for compatibility
    if urg == 0:
        mu = np.tile(mu, (len(tvec), 1))

    elif isinstance(urg, (int, float)) or (len(urg) == 1):
        urg_vec = np.ones(len(tvec)-1) * urg/(len(tvec)-1)
        urg_vec = np.insert(urg_vec, 0, 0)
        mu = mu * urg_vec
    else: # assume vector like tvec
        assert len(urg) == len(tvec), "If urgency signal is a vector, it must match tvec"
        mu = mu * urg

    return mu


def log_odds(pdf1, pdf2):
    """
    calculate log posterior odds of correct choice
    assumes that drift is the first dimension, which gets marginalized over
    :param pdf1: pdf of losing race for correct trials
    :param pdf2: pdf of losing race for incorrect trials
    :return log_odds_correct: heatmap, log posterior odds of correct choice
    """

    pdf1[pdf1 == 0] = np.finfo(np.float64).tiny
    pdf2[pdf2 == 0] = np.finfo(np.float64).tiny

    odds = np.sum(pdf1, axis=0) / np.sum(pdf2, axis=0)
    odds[odds < 1] = 1
    return np.log(odds)

def log_pmap(pdf, q=30):
    """
    for visualization of losing accumulator pdf, it's likely helpful to perform a cutoff and look at log space
    :param pdf:
    :param q:
    :return:
    """
    pdf[pdf < 10**(-q)] = 10**(-q)
    return (np.log10(pdf)+q) / q


def main():

    xmesh, ymesh = np.mgrid[-7:0:0.05, -7:0:0.05]
    xy_mesh = np.dstack((xmesh, ymesh))

    tvec = np.arange(0, 2, 0.005)
    cohs = [0, 0.032, 0.064, 0.128, 0.256, 0.512]

    sensitivity = 15
    num_images = 7
    bound = np.array([1, 1], dtype=float)

    p_up_coh, RTdist, pdf_cohs, p_up_lose_pdf, p_lo_lose_pdf = [], [], [], [], []
    for coh in cohs:
        mu = np.array([sensitivity*coh, sensitivity*-coh], dtype=float)
        mu = urgency_scaling(mu, tvec, urg_max=0)  # no urgency signal

        start = time.time()

        p_up, r = moi_cdf(tvec, mu, bound, num_images)
        pdf_m = moi_pdf(xmesh, ymesh, tvec, mu, bound, num_images)

        end = time.time()
        print(f"Elapsed time = {end-start:.4f}")

        pdf_cohs.append(pdf_m)
        p_up_coh.append(p_up)
        RTdist.append(r)

        # pdfs are time by x*y !
        p_up_lose_pdf.append(np.squeeze(np.sum(pdf_m, axis=2))) # sum over y
        p_lo_lose_pdf.append(np.squeeze(np.sum(pdf_m, axis=1))) # sum over x

    p_up_coh = np.array(p_up_coh)
    RTdist = np.stack(RTdist, axis=0)

    p_up_lose_pdf = np.stack(p_up_lose_pdf, axis=0)
    p_lo_lose_pdf = np.stack(p_lo_lose_pdf, axis=0)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(cohs, p_up_coh)
    ax[2].plot(tvec, RTdist.T)

    log_odds_correct = log_odds(p_up_lose_pdf, p_lo_lose_pdf)


if __name__ == '__main__':
    main()