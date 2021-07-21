import numpy as np
import hmc
import matplotlib.pyplot as plt
import corner


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
nparam = len(limits)

def mock_posterior_and_gradient(p):
    logP = -0.5 * np.sum(p**2)
    logP_jacobian = p * (-1)
    return logP, logP_jacobian

def run_hmc(n_it, filebase, epsilon, steps_per_iteration):
    #rank = 5
    rank = nparam
    filename = f'{filebase}.{rank}.txt'
    np.random.seed(100 + rank)
    C = np.eye(nparam)
    # mass matrix
    sampler = hmc.HMC(mock_posterior_and_gradient, C, epsilon, steps_per_iteration, limits)
    # first sample starts at fid
    fid_params  = np.zeros(nparam)
    results = sampler.sample(n_it, fid_params)

    # continue
    #for i in range(1):
        # Save chain
        #chain = np.array(sampler.trace)
        #np.savetxt(filename, chain)

        # next round of samples
    #sampler.sample(n_it)
    
    chain = np.array(sampler.paths)
    np.savetxt(filename, chain)

nit = 50
spit = 50
run_hmc(nit, "hmc_002_500", 0.02, spit)

chain = np.genfromtxt("hmc_002_500.3.txt")
print(chain.shape)
plt.plot(chain[:,0],chain[:,1])
plt.savefig("plot.png")
figure = corner.corner(chain)
figure.savefig("cornerplot.png")
print("plotted")

