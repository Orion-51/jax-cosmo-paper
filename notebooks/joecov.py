import jax
import os
import jax.numpy as np
import jax_cosmo as jc
import numpy as onp
from desy1 import theory_cov, get_params_vec, get_data

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
nz_source, nz_lens = get_data()

neff_s = [1.47, 1.46, 1.50, 0.73]

nzs_s = [jc.redshift.kde_nz(nz_source['Z_MID'].astype('float32'),
                            nz_source['BIN%d'%i].astype('float32'), 
                            bw=0.01,
                            gals_per_arcmin2=neff_s[i-1])
           for i in range(1,5)]
nzs_l = [jc.redshift.kde_nz(nz_lens['Z_MID'].astype('float32'),
                              nz_lens['BIN%d'%i].astype('float32'), bw=0.01)
           for i in range(1,6)]    

# Define some ell range
ell = np.logspace(1, 3)
args = [nzs_s, nzs_l, ell]
covmat = theory_cov(fid_params, *args)
covmat = onp.array(covmat)
print(covmat)
onp.save("covmat.npy", covmat)
