import jax
import os
import jax.numpy as np
import jax_cosmo as jc
import numpy as onp
from astropy.io import fits
import scipy
from functools import partial


#desy1_v2
nz_source=fits.getdata('2pt_NG_mcal_1110.fits', 6)

neff_s = [1.47, 1.46, 1.50, 0.73]

nzs_s = [jc.redshift.kde_nz(nz_source['Z_MID'].astype('float32'),
                            nz_source['BIN%d'%i].astype('float32'), 
                            bw=0.01,
                            gals_per_arcmin2=neff_s[i-1])
           for i in range(1,5)]

#ell = np.logspace(1, 3)
def get_params_vec(cosmo, m, dz, ia):
    m1, m2, m3, m4 = m
    dz1, dz2, dz3, dz4 = dz
    A, eta = ia
    return np.array([ 
        # Cosmological parameters
        cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b,
        cosmo.h, cosmo.n_s, cosmo.w0,
        # Shear systematics
        m1, m2, m3, m4,
        # Photoz systematics
        dz1, dz2, dz3, dz4,
        # IA model
        A, eta
    ])
    
def unpack_params_vec(params):
    # Retrieve cosmology
    cosmo = jc.Cosmology(sigma8=params[0], Omega_c=params[1], Omega_b=params[2],
                         h=params[3], n_s=params[4], w0=params[5],
                         Omega_k=0., wa=0.)
    m1,m2,m3,m4 = params[6:10]
    dz1,dz2,dz3,dz4 = params[10:14]
    A = params[14]
    eta = params[15]
    return cosmo, [m1,m2,m3,m4], [dz1,dz2,dz3,dz4], [A, eta]

@jax.jit
def cov(params, ell):
    
    cl_signal = mu(params, ell)
    
    # First unpack parameter vector
    cosmo, m, dz, (A, eta) = unpack_params_vec(params) 
    
    # Build source nz with redshift systematic bias
    nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi) 
                for nzi, dzi in zip(nzs_s, dz)]
    
    # Define IA model, z0 is fixed
    b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)
    # Bias for the lenses
    # b = [jc.bias.constant_linear_bias(bi) for bi in bias]
    
    # Define the lensing and number counts probe
    probes = [jc.probes.WeakLensing(nzs_s_sys, 
                                    ia_bias=b_ia,
                                    multiplicative_bias=m),
    ]
             # jc.probes.NumberCounts(nzs_l, b)]
    
    cl_noise = jc.angular_cl.noise_cl(ell, probes)
    
    cov = jc.angular_cl.gaussian_cl_covariance(ell, probes, cl_signal, cl_noise, f_sky=0.25, 
                                               sparse=False)
    
    return cov

@jax.jit
def mu(params, ell):
    # First unpack parameter vector
    cosmo, m, dz, (A, eta) = unpack_params_vec(params) 

    # Build source nz with redshift systematic bias
    nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi) 
                for nzi, dzi in zip(nzs_s, dz)]

    # Define IA model, z0 is fixed
    b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)
    # Bias for the lenses
    # b = [jc.bias.constant_linear_bias(bi) for bi in bias] 

    # Define the lensing and number counts probe
    probes = [jc.probes.WeakLensing(nzs_s_sys, 
                                    ia_bias=b_ia,
                                    multiplicative_bias=m),
             # jc.probes.NumberCounts(nzs_l, b)]
    ]
    cl = jc.angular_cl.angular_cl(cosmo, ell, probes)

    return cl

jacobian = jax.jit(jax.jacfwd(lambda p, ell: mu(p, ell).flatten()))

import scipy

def symmetrized_matrix(U):
    u"""Return a new matrix like `U`, but with upper-triangle elements copied to lower-triangle ones."""
    M = U.copy()
    inds = onp.triu_indices_from(M,k=1)
    M[(inds[1], inds[0])] = M[inds]
    return M



def symmetric_positive_definite_inverse(M):
    u"""Compute the inverse of a symmetric positive definite matrix `M`.

    A :class:`ValueError` will be thrown if the computation cannot be
    completed.

    """
    import scipy.linalg
    U,status = scipy.linalg.lapack.dpotrf(M)
    if status != 0:
        raise ValueError("Non-symmetric positive definite matrix")
    M,status = scipy.linalg.lapack.dpotri(U)
    if status != 0:
        raise ValueError("Error in Cholesky factorization")
    M = symmetrized_matrix(M)
    return M



def priors(p):
    #priors - index, mean, std. dev
    prior_values = [
        (6, 0.012, 0.023),  #m1
        (7, 0.012, 0.023),  #m2
        (8, 0.012, 0.023),  #m3
        (9, 0.012, 0.023),  #m4
        (10, -0.001, 0.016),   #dz1
        (11, -0.019, 0.013),   #dz2
        (12,0.009, 0.011),   #dz3
        (13, -0.018, 0.022),   #dz4
    ]

    logpi = 0.0
    dlogpi_dp = onp.zeros_like(p)
    for i, mu_i, sigma_i in prior_values:
        logpi += -0.5 * (p[i] - mu_i)**2 / sigma_i**2
        dlogpi_dp[i] = - (p[i] - mu_i) / sigma_i**2
    return logpi, dlogpi_dp


def posterior_and_gradient(p, ell, data_mean, inv_cov):
    # theory C_ell prediction
    cl = mu(p, ell).flatten()
    # d C_ell / d p
    j = jacobian(p, ell).T

    d = cl - data_mean
    dlogL_dCl = -inv_cov @ d

    # convert back to regular numpy arrays
    # after calculating
    logL = onp.array(0.5 * d @ dlogL_dCl)
    dlogL_dp = onp.array(j @ dlogL_dCl)

    
    # Add Gaussian priors.
    logPi, dlogPi_dp = priors(p)
    logP = logL + logPi
    dlogP_dp = dlogL_dp + dlogPi_dp

    return logP, dlogP_dp


class MockY1Likelihood:
    def __init__(self):
        fid_cosmo = jc.Cosmology(sigma8=0.801,
                                 Omega_c=0.2545,
                                 Omega_b=0.0485,
                                 h=0.682,
                                 n_s=0.971,
                                 w0=-1., Omega_k=0., wa=0.)

        self.ell = np.logspace(1, 3, 50)

        self.fid_params = get_params_vec(fid_cosmo, 
                                [1.2e-2, 1.2e-2, 1.2e-2, 1.2e-2],
                                [0.1e-2, -1.9e-2, 0.9e-2, -1.8e-2],
                                [0.5, 0.])

        self.data_mean = onp.array(mu(self.fid_params, self.ell).flatten())
        self.inv_cov = onp.array(cov(self.fid_params, self.ell))

    def posterior_and_gradient(self, p):
        return posterior_and_gradient(p, self.ell, self.data_mean, self.inv_cov)

    def posterior_and_gradient_3d(self, p):
        pfull = self.fid_params.copy()
        pfull[0] = p[0]
        pfull[1] = p[1]
        pfull[3] = p[2]
        return posterior_and_gradient(pfull, self.ell, self.data_mean, self.inv_cov)



def main():
    import time
    t0 = time.time()

    calc = MockY1Likelihood()
    print("Time for mean and inv cov: {:.3f}".format(time.time() - t0))


    logLs = []
    for sigma8 in onp.arange(0.75, 0.85, 0.01):
        t0 = time.time()
        cosmo = jc.Cosmology(sigma8=sigma8,
                             Omega_c=0.2545,
                             Omega_b=0.0485,
                             h=0.682,
                             n_s=0.971,
                             w0=-1., Omega_k=0., wa=0.)

        params = get_params_vec(cosmo, 
                            [1.2e-2, 1.2e-2, 1.2e-2, 1.2e-2],
                            [0.1e-2, -1.9e-2, 0.9e-2, -1.8e-2],
                            [0.5, 0.])

        logL, jac = calc.posterior_and_gradient(params)
        logLs.append(logL)
        print("Time for iteration {}: {:.3f}".format(sigma8, time.time() - t0))


if __name__ == '__main__':
    main()
