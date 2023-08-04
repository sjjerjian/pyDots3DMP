import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from numba import njit, prange
import numba as nb
import time
import timeit


#@njit(nb.f8[:](nb.u1, nb.f8[:], nb.u1))
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


#@njit(nb.f8(nb.u1, nb.f8[:], nb.f8[:, :], nb.f8[:], nb.f8[:]))
def weightj(j, mu, sigma, sj, s0):

    return (-1) ** j * np.exp(mu @ np.linalg.inv(sigma) @ (sj - s0).T)

    # dot_prod = np.dot(sj - s0, np.linalg.inv(sigma))
    # dot_prod = np.ascontiguousarray(dot_prod)
    #
    # explicit dot product calculation
    # result = 0.0
    # for i in range(len(mu)):
    #     result += mu[i] * dot_prod[i]
    #
    # return (-1) ** j * np.exp(result)

    #return (-1) ** j * np.exp(np.dot(mu, dot_prod))

@njit(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :]))
def mv_pdf(x_mesh, y_mesh, mean, covariance):

    # this doesn't give the same results as scipy.pdf right now...need to understand why
    k = len(mean)
    inv_cov = np.linalg.inv(covariance)

    pdf_values = np.zeros((x_mesh.shape[0], y_mesh.shape[1]))

    for i in range(x_mesh.shape[0]):
        for j in range(y_mesh.shape[1]):
            diff0 = x_mesh[i, j] - mean[0]
            diff1 = y_mesh[i, j] - mean[1]

            dot_product = 0.0
            for ik in range(k):
                dot_product += diff0 * inv_cov[ik, 0] * diff0 + \
                               diff1 * inv_cov[ik, 1] * diff1

            exponent = -0.5 * dot_product
            coefficient = 1.0 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(covariance)))
            pdf_values[i, j] = coefficient * np.exp(exponent)

    return pdf_values


#@njit(nb.f8[:, :, :](nb.f8[:, :, :], nb.f8[:], nb.f8[:, :], nb.f8[:], nb.u1), parallel=True)
def moi_pdf(xy_mesh, tvec: np.ndarray, mu: np.ndarray, s0: np.ndarray, num_images: int):

    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    pdf_result = np.zeros((xy_mesh.shape[0], xy_mesh.shape[1], len(tvec)))

    for t in range(1, len(tvec)):
        ti = tvec[t]

        mu_t = mu[t, :]

        # 1. hand-calculate pdfs, for numba
        mvn_lhs_pdf = mv_pdf(xy_mesh[:, :, 0], xy_mesh[:, :, 1],
                             s0 + mu_t * ti, sigma * ti)

        mvn_rhs_pdf = np.zeros((mvn_lhs_pdf.shape[0], mvn_lhs_pdf.shape[1], (k*2)-1))
        for j in range(1, k*2):
            mvn_rhs_pdf[:, :, j-1] = weightj(j, mu_t, sigma, sj_rot(j, s0, k), s0) * \
                mv_pdf(xy_mesh[:, :, 0], xy_mesh[:, :, 1],
                       sj_rot(j, s0, k) + mu_t * ti, sigma * ti)
        mvn_rhs_pdf = mvn_rhs_pdf.sum(axis=2)  # average across weighted images

        # 2. using scipy.multivariate_normal, probably can't use with numba
        mvn_lhs_pdf2 = mvn(mean=s0 + mu_t * ti, cov=sigma * ti).pdf(xy_mesh)

        mvn_rhs_pdf2 = np.sum([
            weightj(j, mu_t, sigma, sj_rot(j, s0, k), s0) * \
            mvn(mean=sj_rot(j, s0, k) + mu_t * ti, cov=sigma * ti).pdf(xy_mesh)
            for j in range(1, k * 2)
        ])

        # equivalent to 2., but with explicit loop for rhs
        # mvn_rhs_pdf = 0
        # for j in range(1, k*2):
        #     sj = sj_rot(j, s0, k)
        #     a_j = weightj(j, mu_t, sigma, sj, s0)
        #     mvn_rhs_pdf += a_j * mvn(mean=sj + mu_t*ti, cov=sigma*ti).pdf(xy_mesh)

        pdf_result[:, :, t] = mvn_lhs_pdf + mvn_rhs_pdf

    return pdf_result


def moi_cdf(tvec, mu, s0=np.array([-10, -10]), num_images=7, bound=-0.025, bound_marginal=0):

    k = int(np.ceil(num_images / 2))
    rho = -np.cos(np.pi / k)
    sigma = np.array([[1, rho], [rho, 1]])

    survival_prob = np.ones(len(tvec))
    flux1 = np.zeros_like(survival_prob)
    flux2 = np.zeros_like(survival_prob)
    #moi_weights = np.zeros((len(tvec), k*2-1))

    bound0 = np.array([bound, bound])
    bound2 = np.array([bound_marginal, bound])
    bound1 = np.array([bound, bound_marginal])

    # skip the first sample (t starts at 1)
    for t in range(1, len(tvec)):

        mu_t = mu[t, :].T * tvec[t]

        mvn_0 = mvn(s0 + mu_t, cov=sigma * tvec[t])

        # total density within boundaries
        cdf_rest = mvn_0.cdf(bound0)

        # density beyond boundary in one or other direction
        cdf1 = mvn_0.cdf(bound1) - cdf_rest
        cdf2 = mvn_0.cdf(bound2) - cdf_rest

        for j in range(1, k*2):
            sj = sj_rot(j, s0, k)

            mvn_j = mvn(sj + mu_t, cov=sigma*tvec[t])

            # total density within boundaries
            cdf_add = mvn_j.cdf(bound0)

            # density outside of boundary in one or other direction
            cdf_add1 = mvn_j.cdf(bound1) - cdf_add
            cdf_add2 = mvn_j.cdf(bound2) - cdf_add

            a_j = weightj(j, mu[t, :].T, sigma, sj, s0)
            #moi_weights[t, j-1] = a_j

            cdf_rest += (a_j * cdf_add)
            cdf1 += (a_j * cdf_add1)
            cdf2 += (a_j * cdf_add2)

        survival_prob[t] = cdf_rest
        flux1[t] = cdf1
        flux2[t] = cdf2

    p_up = np.sum(flux2) / np.sum(flux1 + flux2)

    rt_dist = np.diff(np.insert(1-survival_prob, 0, 0))

    return p_up, rt_dist


if __name__ == '__main__':


    xmesh, ymesh = np.mgrid[-5*2:3:0.1, -5*2:3:0.1]
    xy_mesh = np.dstack((xmesh, ymesh))

    tvec = np.arange(0, 3, 0.01)

    #cohs = [0, 0.032, 0.064, 0.128, 0.256, 0.512]
    cohs = [0]
    sensitivity = 3

    num_images = 7
    s0 = np.array([-1, -1], dtype=float)
    s0 = np.ascontiguousarray(s0)
    bound = 0

    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])

    p_up_coh, RTdist, pdf_cohs = [], [], []

    for coh in cohs:
        mu = np.array([sensitivity*coh, sensitivity*-coh], dtype=float)
        mu = np.tile(mu, (len(tvec), 1))  # here we would incorporate urgency signal

        start = time.time()
        #pdfs = mv_pdf(xmesh, ymesh, mu[0, :], sigma)
        #p_up, r = moi_cdf(tvec, mu, s0, num_images)
        pdf_m = moi_pdf(xy_mesh, tvec, mu, s0, num_images)
        end = time.time()

        print(f"Elapsed time = {end-start:.4f}")
        #pdf_cohs.append(pdf_m)
        #p_up_coh.append(p_up)
        #RTdist.append(r)

    #p_up_coh = np.array(p_up_coh)
    #RTdist = np.vstack(RTdist)
    #plt.plot(RTdist.T)

    mean = np.array([1.0, 2.0])
    # covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    # xy_mesh = np.random.randn(100, 2)  # Example mesh
    #
    # pdf_values = mv_pdf(xy_mesh, mean, covariance)

