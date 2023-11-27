import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from numba import njit, prange
import numba as nb
import time
import timeit


#@njit(nb.f8[:](nb.u1, nb.f8[:], nb.u1), fastmath=True)
def sj_rot(j, s0, k):
    """
    Image rotation formalism
    """

    alpha = (k - 1)/k * np.pi
    sin_alpha = np.sin(j * alpha)
    sin_alpha_plus_k = np.sin(j * alpha + np.pi / k)
    sin_alpha_minus_k = np.sin(j * alpha - np.pi / k)

    if j % 2 == 0:
        s = np.array([[sin_alpha_plus_k, sin_alpha], [-sin_alpha, -sin_alpha_minus_k]])
    else:
        s = np.array([[sin_alpha, sin_alpha_minus_k], [-sin_alpha_plus_k, -sin_alpha]])

    # result = np.zeros_like(s0)
    # for i in range(2):
    #     for j in range(2):
    #         result[i] += s[i, j] * s0[j]
    # return (1 / np.sin(np.pi / k)) * result

    return (1 / np.sin(np.pi / k)) * (s @ s0.T)


# @njit(nb.f8(nb.u1, nb.f8[:], nb.f8[:, :], nb.f8[:], nb.f8[:]), fastmath=True)
def weightj(j, mu, sigma, sj, s0):

    return (-1) ** j * np.exp(mu @ np.linalg.inv(sigma) @ (sj - s0).T)

    # explicit dot product calculation
    # dot_prod = np.dot(sj - s0, np.linalg.inv(sigma))
    # result = 0.0
    # for i in range(len(mu)):
    #     result += mu[i] * dot_prod[i]
    #
    # return (-1) ** j * np.exp(result)

@njit(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :]), parallel=True, fastmath=True)
def mv_pdf(x_mesh, y_mesh, mean, covariance):

    k = len(mean)
    inv_cov = np.linalg.inv(covariance)

    pdf_values = np.empty((x_mesh.shape[0], y_mesh.shape[1]))
    for i in prange(x_mesh.shape[0]):
        for j in prange(y_mesh.shape[1]):
            diff = np.array([x_mesh[i, j] - mean[0], y_mesh[i, j] - mean[1]])

            exponent = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
            coefficient = 1.0 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(inv_cov)))

            pdf_values[i, j] = coefficient * np.exp(exponent)

    return pdf_values


# @njit(nb.f8[:, :, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :], nb.f8[:], nb.u1), parallel=True, fastmath=True)
def moi_pdf(xmesh: np.ndarray, ymesh: np.ndarray, tvec: np.ndarray, mu: np.ndarray, s0=np.array([-5, -5]), num_images: int=7):

    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    nx, ny = xmesh.shape
    pdf_result = np.empty((len(tvec), nx, ny))

    #xy_mesh = np.dstack((xmesh, ymesh))

    # skip the first sample (t starts at 1)
    for t in prange(1, len(tvec)):

        # start = time.time()

        mu_t = mu[t, :] * tvec[t]
        sigma_t = sigma * tvec[t]

        # 1. hand-calculate pdfs, for numba
        # pdf_result[t, :, :] = mv_pdf(xmesh, ymesh, s0 + mu_t, sigma_t)
        #
        # for j in range(1, k*2):
        #     sj = sj_rot(j, s0, k)
        #
        #     a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
        #     pdf_result[t, :, :] += a_j * mv_pdf(xmesh, ymesh, sj + mu_t, sigma_t)


        # 2. using scipy.multivariate_normal, can't use with numba
        pdf_result[t, :, :] = mvn(mean=s0 + mu_t, cov=sigma_t).pdf(xy_mesh)
        for j in range(1, k*2):
            sj = sj_rot(j, s0, k)
            a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
            pdf_result[t, :, :] += a_j * mvn(mean=sj + mu_t, cov=sigma_t).pdf(xy_mesh)

        # end = time.time()
        # if t%10 == 0:
        #    print(f"Elapsed time, timestep {t} = {end-start:.4f}")

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


def moi_cdf(tvec, mu, s0=np.array([-5, -5]), num_images: int=7, bound=-0.025, bound_marginal=0):

    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    survival_prob = np.ones(len(tvec))
    flux1, flux2 = np.empty(len(tvec)), np.empty(len(tvec))

    bound0 = np.array([bound, bound])
    bound1 = np.array([bound, bound_marginal])  # top boundary of third quadrant
    bound2 = np.array([bound_marginal, bound])  # right boundary

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

            # total density within boundaries for jth image
            cdf_add = mvn_j.cdf(bound0)

            # density outside of boundary in one or other direction, for jth image
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

    rt_dist = np.diff(np.insert(1-survival_prob, 0, 0))

    # winning and losing pdfs?
    pdf_up = np.diff(flux2)
    pdf_lo = np.diff(flux1)

    return p_up, rt_dist


def urgency_scaling(mu, tvec, urg_max=0):
    if urg_max == 0:
        mu = np.tile(mu, (len(tvec), 1))
    else:
        urg_vec = np.ones(len(tvec)-1) * urg_max/(len(tvec)-1)
        urg_vec = np.insert(urg_vec, 0, 0)
        mu = mu * urg_vec

    return mu


def images_dtb_calc(xy_mesh, tvec, mu, s0=np.array([-5, -5]), num_images: int=7, bound=-0.025, bound_marginal=0):

    # should also store the tvec and other parameters
    # could maybe make this a dataclass, an images_dtb object

    # but maybe we want to concatenate cohs first...

    P = dict()
    P['tvec'] = tvec
    P['pdf'] = moi_pdf(xy_mesh[:, :, 0], xy_mesh[:, :, 0], tvec, mu, s0, num_images)
    P['p_up'], P['rt_dist'] = moi_cdf(tvec, mu, s0, num_images, bound, bound_marginal)

    return P



def log_odds(pdf1, pdf2):
    """
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

    pdf[pdf < 10**(-q)] = 10**(-q)
    return (np.log10(pdf)+q) / q


if __name__ == '__main__':

    xmesh, ymesh = np.mgrid[-7:0:0.05, -7:0:0.05]
    xy_mesh = np.dstack((xmesh, ymesh))

    tvec = np.arange(0, 2, 0.005)

    cohs = [0, 0.032, 0.064, 0.128, 0.256, 0.512]
    sensitivity = 15

    num_images = 7
    s0 = np.array([-3, -3], dtype=float)
    bound = 0

    #sigma = np.array([[1.0, -0.71], [-0.71, 1.0]])
    p_up_coh, RTdist, pdf_cohs, \
        p_up_lose_pdf, p_lo_lose_pdf = [], [], [], [], []

    for coh in cohs:
        mu = np.array([sensitivity*coh, sensitivity*-coh], dtype=float)
        mu = urgency_scaling(mu, tvec, urg_max=0)  # no urgency signal

        start = time.time()
        p_up, r = moi_cdf(tvec, mu, s0, num_images)
        pdf_m = moi_pdf(xmesh, ymesh, tvec, mu, s0, num_images)

        #P = images_dtb_calc(xy_mesh, tvec, mu, s0, num_images)

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

    odds = np.sum(p_up_lose_pdf, axis=0) / np.sum(p_lo_lose_pdf, axis=0)
    odds[odds < 1] = 1
    log_odds_correct = log_odds(p_up_lose_pdf, p_lo_lose_pdf)




    #mean = np.array([1.0, 2.0])
    # covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    # xy_mesh = np.random.randn(100, 2)  # Example mesh
    #
    # pdf_values = mv_pdf(xy_mesh, mean, covariance)

