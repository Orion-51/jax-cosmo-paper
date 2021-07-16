import numpy as np
import hmc
import jax_cosmo as jc


limits = [
    (0.5, 0.9), # sigma8
    (0.1, 0.5), # Omega_c
    (0.03, 0.06), # Omega_b
    (0.5,  0.9), # h
    (0.9,  1.05), # n_s
    (-2.0,  -0.5), # w0
    (-0.06, 0.06), #m1
    (-0.06, 0.06), #m2
    (-0.06, 0.06), #m3
    (-0.06, 0.06), #m4
    (-0.1, 0.1), #dz1
    (-0.1, 0.1), #dz2
    (-0.1, 0.1), #dz3
    (-0.1, 0.1), #dz4
    (0.0, 3.0),  #A
    (-3., 3.), #eta
    (0.8, 3.0), #bias1
    (0.8, 3.0), #bias2
    (0.8, 3.0), #bias3
    (0.8, 3.0), #bias4
    (0.8, 3.0), #bias5
]

# First, let's define a function to go to and from a 1d parameter vector
def get_params_vec(cosmo, m, dz, ia, bias):
    m1, m2, m3, m4 = m
    dz1, dz2, dz3, dz4 = dz
    A, eta = ia
    b1, b2, b3, b4, b5 = bias
    return np.array([ 
        # Cosmological parameters
        cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b,
        cosmo.h, cosmo.n_s, cosmo.w0,
        # Shear systematics
        m1, m2, m3, m4,
        # Photoz systematics
        dz1, dz2, dz3, dz4,
        # IA model
        A, eta,
        # linear galaxy bias
        b1, b2, b3, b4, b5
    ])

def mock_posterior_and_gradient(p):
    logP = -0.5 * np.sum(p**2)
    logP_jacobian = p * (-1)
    return logP, logP_jacobian

def run_hmc(n_it, filebase, epsilon, steps_per_iteration):
    rank = 2
    filename = f'{filebase}.{rank}.txt'
    np.random.seed(100 + rank)
    C = np.eye(len(p))
    # mass matrix
    sampler = hmc.HMC(mock_posterior_and_gradient, C, epsilon, steps_per_iteration, limits)
    # first sample starts at fid
    fid_cosmo = jc.Cosmology(sigma8=0.801,
                                  Omega_c=0.2545,
                                  Omega_b=0.0485,
                                  h=0.682,
                                  n_s=0.971,
                                  w0=-1., Omega_k=0., wa=0.)
    fid_params  = get_params_vec(fid_cosmo, 
                                          [0., 0., 0., 0.],
                                          [0., 0., 0., 0.],
                                          [0.5, 0.],
                                          [1.2, 1.4, 1.6, 1.8, 2.0])
    results = sampler.sample(n_it, fid_params)

    # continue
    for i in range(1000):
        # Save chain
        chain = np.array(sampler.trace)
        np.savetxt(filename, chain)

        # next round of samples
        sampler.sample(n_it)

run_hmc(3, "hmc_002_25", 0.02, 25)