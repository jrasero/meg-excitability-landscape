"""

Analyze association of measurements maps with E/I neurotransmitters
(e.g. GABA, Glutamate) to better understand the function of each
cluster

"""

# ================================
# 1. Imports
# ================================

import numpy as np
from scipy.stats import rankdata
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import load_data, get_region_labels
from plots import plot_parcellated_data

from netneurotools import stats
from scipy.stats import zscore
from nilearn.datasets import fetch_atlas_schaefer_2018
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle


# ================================
# 2. Load Data
# ================================

labels = get_region_labels()
data_mats, measures, _ = load_data()

# stack matrices
X = np.array(list(data_mats.values()))
# Average over subjects
X_group = X.mean(axis=0)

# Load cluster IDs
clus_id = np.load(
    "/home/javi/Documentos/meg-excitability-landscape/data/clusters_new.npz")["clus_id"]

# Take the ranks along each measure, and average over measures in the same cluster
X_clus = np.array([rankdata(X_group, axis=1)[clus_id==label,:].mean(axis=0) \
                   for label in np.unique(clus_id)])
# Just arrange the measures as columns, and regions as observations
X_clus = np.swapaxes(X_clus, 0, 1)

n_regs = X_clus.shape[0]
assert n_regs == 100

n_clus = X_clus.shape[-1]
assert n_clus == 6

# ----- Receptors data -----

scale = 'scale100'
receptor_data = np.genfromtxt(
    '/home/javi/Documentos/meg-excitability-landscape/data/receptors/scale100/receptor_data_'+scale+'.csv', 
    delimiter=',')
receptor_names = np.load(
    '/home/javi/Documentos/meg-excitability-landscape/data/receptors/scale100/receptor_names_pet.npy')

# Take only those receptors related to E/I
ei_receptors = ["A4B2", "M1", "VAChT", "NMDA", "mGluR5", "GABAa" ]

ei_idxs = np.concatenate([np.where(receptor_names==label)[0] 
                          for label in ei_receptors])

# Take the ranks -- we are always using spearman correlations
ei_data = rankdata(receptor_data[:, ei_idxs], axis=0)

# ----- Coordinates, for spatial null maps -----
schaefer = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
coords = pd.read_csv(
    "https://raw.githubusercontent.com/ThomasYeoLab/"+\
        "CBIG/refs/heads/master/stable_projects/brain_parcellation/"+\
            "Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/"+\
                "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
            )
coords.index = coords.iloc[:,1]
# Order according to hansen's repo, just in case
coords = coords.loc[schaefer["labels"].astype(str),:]

# ----- Spatially rearrange data -----
# All dhese data are not spatially arranged as our measurements maps,
# so we have to make sure thet match region by region

# Step 1: Create a mapping from string to its order in `b`
order = {val: i for i, val in enumerate(labels)}

# Step 2: Get the indices that would sort `a` according to `b`
indices = np.argsort([order[val] for val in schaefer["labels"].astype(str)])

# Step 3: Apply these indices to both coordinates and receptors data
coords_ordered = coords.iloc[indices,:]
ei_data_ordered = ei_data[indices, :]

pheno_data = ei_data_ordered.copy()

# ================================
# 3. Analysis
# ================================

# Step 1: Load spins samples
spins = np.load("/home/javi/Documentos/meg-excitability-landscape/data/spin_permutations.npy")
nspins = spins.shape[-1]


# Step 2: correlate each measurement with receptors data
# Since we are inputting the ranked data, that means that pearson correlation
# corresponds to spearman

pvals = np.zeros((X_clus.shape[-1], 
                  pheno_data.shape[-1]))
for ii in range(X_clus.shape[-1]):
    for jj in range(pheno_data.shape[-1]):
        pvals[ii, jj] = stats.permtest_pearsonr(X_clus[:,ii], 
                                                pheno_data[:,jj],
                                                n_perm=nspins,
                                                resamples=spins, seed=1234)[1]

# Flatten the matrix to apply FDR correction
pvals_flat = pvals.flatten()

# Apply FDR correction 
rejected, pvals_corrected, _, _ = multipletests(pvals_flat, 
                                                alpha=0.05, 
                                                method='fdr_bh')

# Reshape corrected p-values back to 5x6
pvals_corrected_matrix = pvals_corrected.reshape(pvals.shape)
rejected_matrix = rejected.reshape(pvals.shape)

# Output
print("Corrected p-values:\n", pvals_corrected_matrix)
print("Rejected hypotheses (FDR controlled):\n", rejected_matrix)

# ================================
# 4. PLOTS
# ================================

# Plot brainplots for clusters
clus_names = ["40Hz", "Alpha", "DFA", "Exponent", "Phase_Ex","SE"]
for ii, title in zip(range(X_clus.shape[-1]), 
                     clus_names):
    
    values_dict = {}
    for key, value in zip(labels, X_clus[:,ii]):
        values_dict[key] = value

    fig = plot_parcellated_data(values_dict, cmap="plasma")
    fig.axes[0].set_title(title, size=20)
    fig.axes[1].tick_params(labelsize=12)
    fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
    ticks = fig.axes[1].get_xticks()
    fig.axes[1].set_xticks([ticks[0], ticks[-1]])
    fig.axes[1].set_xticklabels(["min", "max"])
    fig.savefig(f"/home/javi/Documentos/meg-excitability-clustering/plots/brainplot_cluster_{ii}.png", 
                dpi=300, bbox_inches="tight")



# Brainplot for recepts
for receptor, dat in zip(ei_receptors, receptor_data[indices,:][:,ei_idxs].T):
    
    data_reg  = {}
    for key, value in zip(labels, dat):
        data_reg[key] = value
        
    fig = plot_parcellated_data(data_reg,cmap="plasma")
    fig.axes[0].set_title(receptor, size=20)
    fig.axes[1].tick_params(labelsize=12)
    fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
    fig.savefig(f"/home/javi/Documentos/meg-excitability-clustering/plots/brainplot_{receptor}.png", 
                dpi=300, bbox_inches="tight")


# Heatmap with correlations
corrs_mat = (zscore(ei_data_ordered).T.dot(zscore(X_clus))/100).T
vmax = abs(corrs_mat).max()
vmin = -vmax
plt.figure(figsize=(8,6))
plt.imshow(corrs_mat, cmap=plt.cm.coolwarm, vmax=vmax, vmin=vmin)
plt.yticks(range(6), clus_names, 
           rotation=0, size=20)
plt.xticks(range(len(ei_receptors)), ei_receptors, rotation=90, size=20)
#plt.xticks(range(6), ei_receptors, rotation=90)
plt.colorbar(fraction=0.046, pad=0.04)
#Overlay asterisks on significant cells
ax = plt.gca()  # get current axis
for i in range(pvals.shape[0]):
    for j in range(pvals.shape[1]):
        
        plt.text(j , i , f"{corrs_mat[i,j]:.2f}", 
                 color='black', ha='center', va='center', fontsize=15)

        if rejected_matrix[i, j]:
            plt.text(j , i - 0.25 , "*", 
                     color='black', ha='center', va='center', fontsize=20)
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                             linewidth=2., edgecolor='black', 
                             facecolor='none', zorder=20)
            ax.add_patch(rect)

plt.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/heatmap_receptors_clusters.png", 
            dpi=300, bbox_inches="tight")
plt.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/heatmap_receptors_clusters.svg", 
            dpi=300, bbox_inches="tight")