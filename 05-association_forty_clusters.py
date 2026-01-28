"""
Spatial association between 40Hz and the rest of clusters

"""

# ================================
# 1. Imports
# ================================

import numpy as np
from scipy.stats import rankdata
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-clustering/src")
from input_data import load_data, get_region_labels
from netneurotools import stats

# ================================
# 2. Load Data
# ================================

labels = get_region_labels()
psd_mats, strategies = load_data()

# stack matrices
X = np.array(list(psd_mats.values()))
# Average over sessions
X_group = X.mean(axis=0)

# Load cluster IDs
clus_id, clus_labels = np.load(
    "/home/javi/Documentos/meg-excitability-clustering/data/clusters_new.npz").values()

# Take the ranks along each measure, and average over measures in the same cluster
X_clus = np.array([rankdata(X_group, axis=1)[clus_id==label,:].mean(axis=0) \
                   for label in np.unique(clus_id)])
# Just arrange the measures as columns, and regions as observations
X_clus = np.swapaxes(X_clus, 0, 1)

n_regs = X_clus.shape[0]
assert n_regs == 100

n_clus = X_clus.shape[-1]
assert n_clus == 6

spins = np.load("/home/javi/Documentos/meg-excitability-clustering/data/spin_permutations.npy")
nspins = spins.shape[-1]


clus_names = [clus_labels[clus_id==label][0] for label in np.unique(clus_id)]

# =================================
# 3. Perform spatial correlations #
# =================================


pvals = np.zeros(n_clus-1)
r_vals = np.ones(n_clus-1)

for ii in range(1, n_clus):
    
    r, p = stats.permtest_pearsonr(X_clus[:,0], 
                                   X_clus[:, ii], n_perm=nspins,
                                   resamples=spins, seed=1234)
    pvals[int(ii-1)] = p
    r_vals[int(ii-1)] = r
    print(clus_names[0], clus_names[ii], r, p)

