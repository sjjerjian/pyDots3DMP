import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from numba import njit, prange
import numba as nb
import time
import timeit


@njit(nb.f8[:](nb.u1, nb.f8[:], nb.u1), fastmath=True)
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

    result = np.zeros_like(s0)
    for i in range(2):
        for j in range(2):
            result[i] += s[i, j] * s0[j]
    return (1 / np.sin(np.pi / k)) * result

    # return (1 / np.sin(np.pi / k)) * (s @ s0.T)


@njit(nb.f8(nb.u1, nb.f8[:], nb.f8[:, :], nb.f8[:], nb.f8[:]), fastmath=True)
def weightj(j, mu, sigma, sj, s0):

    # return (-1) ** j * np.exp(mu @ np.linalg.inv(sigma) @ (sj - s0).T)

    # explicit dot product calculation
    dot_prod = np.dot(sj - s0, np.linalg.inv(sigma))
    result = 0.0
    for i in range(len(mu)):
        result += mu[i] * dot_prod[i]

    return (-1) ** j * np.exp(result)

@njit(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :]), parallel=True, fastmath=True)
def mv_pdf(x_mesh, y_mesh, mean, covariance):

    k = len(mean)
    inv_cov = np.linalg.inv(covariance)

    pdf_values = np.empty((x_mesh.shape[0], y_mesh.shape[1]))
    for i in prange(x_mesh.shape[0]):
        for j in range(y_mesh.shape[1]):
            diff = np.array([x_mesh[i, j] - mean[0], y_mesh[i, j] - mean[1]])

            exponent = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
            coefficient = 1.0 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(inv_cov)))

            pdf_values[i, j] = coefficient * np.exp(exponent)

    return pdf_values


@njit(nb.f8[:, :, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :], nb.f8[:], nb.u1), fastmath=True)
def moi_pdf(xmesh: np.ndarray, ymesh: np.ndarray, tvec: np.ndarray, mu: np.ndarray, s0: np.ndarray, num_images: int):

    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    nx, ny = xmesh.shape
    pdf_result = np.empty((len(tvec), nx, ny))

    # skip the first sample (t starts at 1)

    for t in range(1, len(tvec)):
        mu_t = mu[t, :] * tvec[t]
        sigma_t = sigma * tvec[t]

        # 1. hand-calculate pdfs, for numba
        pdf_result[t, :, :] = mv_pdf(xmesh, ymesh, s0 + mu_t, sigma_t)

        for j in range(1, k*2):
            sj = sj_rot(j, s0, k)

            a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
            pdf_result[t, :, :] += a_j * mv_pdf(xmesh, ymesh, sj + mu_t, sigma_t)

        # 2. using scipy.multivariate_normal, can't use with numba
        # pdf_result[t, :, :] = mvn(mean=s0 + mu_t, cov=sigma*tvec[t]).pdf(xy_mesh)

        # equivalent to above, but with explicit loop for rhs
        # for j in range(1, k*2):
        #     sj = sj_rot(j, s0, k)
        #     a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
        #     pdf_result[t, :, :] += a_j * mvn(mean=sj + mu_t, cov=sigma_t).pdf(xy_mesh)

    return pdf_result


def moi_cdf(tvec, mu, s0=np.array([-5, -5]), num_images=7, bound=-0.025, bound_marginal=0):

    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    survival_prob = np.empty(len(tvec))
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


if __name__ == '__main__':

    xmesh, ymesh = np.mgrid[-10:0:0.05, -10:0:0.05]
    xy_mesh = np.dstack((xmesh, ymesh))

    tvec = np.arange(0, 2, 0.01)

    cohs = [0, 0.032, 0.064, 0.128, 0.256, 0.512]
    sensitivity = 5

    num_images = 7
    s0 = np.array([-5, -5], dtype=float)
    bound = 0

    sigma = np.array([[1.0, -0.71], [-0.71, 1.0]])
    p_up_coh, RTdist, pdf_cohs, \
        p_up_lose_pdf, p_lo_lose_pdf = [], [], [], [], []

    for coh in cohs:
        mu = np.array([sensitivity*coh, sensitivity*-coh], dtype=float)
        mu = urgency_scaling(mu, tvec, urg_max=0)  # no urgency signal

        start = time.time()
        #pdfs = mv_pdf(xmesh, ymesh, mu[0, :], sigma)
        p_up, r = moi_cdf(tvec, mu, s0, num_images)
        pdf_m = moi_pdf(xmesh, ymesh, tvec, mu, s0, num_images)
        end = time.time()

        print(f"Elapsed time = {end-start:.4f}")
        pdf_cohs.append(pdf_m)
        p_up_coh.append(p_up)
        RTdist.append(r)

        p_up_lose_pdf.append(np.squeeze(np.sum(pdf_m, axis=1)))
        p_lo_lose_pdf.append(np.squeeze(np.sum(pdf_m, axis=0)))

    p_up_coh = np.array(p_up_coh)
    RTdist = np.vstack(RTdist)
    plt.plot(RTdist.T)

    #mean = np.array([1.0, 2.0])
    # covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    # xy_mesh = np.random.randn(100, 2)  # Example mesh
    #
    # pdf_values = mv_pdf(xy_mesh, mean, covariance)

