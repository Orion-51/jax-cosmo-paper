import numpy as np
import hmc
import matplotlib.pyplot as plt


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

limits = [(-4,4),(-4,4),(-4,4)]

def mock_posterior_and_gradient(p):
    logP = -0.5 * np.sum(p**2)
    logP_jacobian = p * (-1)
    return logP, logP_jacobian

def run_hmc(n_it, filebase, epsilon, steps_per_iteration):
    rank = 3
    filename = f'{filebase}.{rank}.txt'
    np.random.seed(100 + rank)
    C = np.eye(len(limits))
    # mass matrix
    sampler = hmc.HMC(mock_posterior_and_gradient, C, epsilon, steps_per_iteration, limits)
    # first sample starts at fid
    fid_params  = np.zeros(len(limits))
    results = sampler.sample(n_it, fid_params)

    # continue
    for i in range(1000):
        # Save chain
        chain = np.array(sampler.trace)
        np.savetxt(filename, chain)

        # next round of samples
        sampler.sample(n_it)

#run_hmc(10, "hmc_002_10", 0.02, 10)

chain = np.genfromtxt("hmc_002_10.3.txt")
print(chain.shape)
plt.plot(chain[:,:1])
plt.savefig("plot.png")
print("plotted")

