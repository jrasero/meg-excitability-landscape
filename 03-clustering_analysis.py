"""
Clustering analysis

"""
# ================================
# 1. Imports
# ================================

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-clustering/src")
from input_data import load_data, get_region_labels
import matplotlib.patches as patches
from sklearn.metrics import silhouette_samples, silhouette_score
from itertools import combinations
from os.path import join as opj

def mean_intercluster_distance(distance_matrix, labels, agg='mean'):
    """
    Compute the mean inter-cluster distance from a precomputed distance matrix and cluster labels.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Precomputed symmetric distance matrix.
    labels : ndarray of shape (n_samples,)
        Cluster labels (ints or strings).
    agg : str
        Aggregation function for pairwise cluster distances: 'mean', 'min', or 'median'.

    Returns
    -------
    score : float
        Mean inter-cluster distance aggregated over all cluster pairs.
        Returns np.nan if fewer than 2 clusters.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    inter_dists = []

    for c1, c2 in combinations(unique_labels, 2):
        idx1 = np.where(labels == c1)[0]
        idx2 = np.where(labels == c2)[0]
        dists = distance_matrix[np.ix_(idx1, idx2)]
        if agg == 'mean':
            inter_dists.append(np.mean(dists))
        elif agg == 'min':
            inter_dists.append(np.min(dists))
        elif agg == 'median':
            inter_dists.append(np.median(dists))
        else:
            raise ValueError("agg must be one of 'mean', 'min', or 'median'")

    return np.mean(inter_dists)


# ================================
# 2. Load Data
# ================================

# ----- data -----

labels = get_region_labels()
data_mats, measures, _ = load_data()

R_scans = []
for key,value in data_mats.items():
    R = pd.DataFrame(value).T.corr("spearman").to_numpy()
    R_scans.append(R)

R_scans = np.array(R_scans)
R_avg = np.array(R_scans).mean(0)

D_scans = 1 - R_scans # Distance matrix by subject
D_avg = 1 - R_avg # Distance matrix for the group


# ================================
# 3. Clustering
# ================================

# Cluster the group based distance matrix
Z = hierarchy.linkage(squareform(D_avg, checks=False), 'average')

link_cols = {10: 'C3', 
             11: 'C9', 
             12: 'C6', 
             13: 'C9',
             14: 'C0',
             15: 'C0',
             16: 'C0',
             17: 'C0',
             18: 'C4'}

outdir= "/home/javi/Documentos/meg-excitability-landscape/plots"
fig, axs = plt.subplots(nrows=2, figsize=(10,10))

# Plot dendogram
dn = hierarchy.dendrogram(Z, 
                          labels=measures, 
                          color_threshold =0.,
                          #link_color_func=lambda x: link_cols[x],
                          above_threshold_color="xkcd:dark grey",
                          ax=axs[0])

ticklabels = dn["ivl" ] 

# Rename some of the measures
for ii, name in enumerate(ticklabels):
    if name == "DFA_EI":
        ticklabels[ii] = "DFA"
    elif name == "Phase_Ex":
        ticklabels[ii] = "EI"
      
axs[0].set_xticklabels("")
axs[0].tick_params(labelsize=0, size=0, length=0, width=0, colors="w")
# Increase the line width
for collection in axs[0].collections:
    collection.set_linewidth(2)  # Set to your desired width

# And the actual correlation matrix below
im = axs[1].imshow(R_avg[dn["leaves"],:][:, dn["leaves"]], aspect="auto")

square_positions = [(0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]
square_size = [2, 1, 1, 1, 2, 3]
square_colors = ["C3", "C2", "C5", "C6", "C4", "C1"]

for (x,y), sq, c in zip(square_positions, square_size, square_colors):
    rect = patches.Rectangle(
            (x-0.5, y-0.5), sq, sq,
            linewidth=5, edgecolor=c, facecolor='none', linestyle="--",
            zorder=20,  # ensures it's on top

        )
    axs[1].add_patch(rect)

axs[1].set_xticks(np.arange(10))
axs[1].set_xticklabels(ticklabels, rotation=45)
axs[1].tick_params(labelsize=20, axis="x")
axs[1].tick_params(labelsize=0, axis="y")
plt.tight_layout()
plt.savefig(opj(outdir, "AllExcitability_dendogram_and_similaritymap_new.png"), 
             dpi=300)


# Calculate average silhouette score and inter-cluster distance 
# for each solution
sil_scores = []
cluster_solutions = np.array([2,3,4,5,6,7, 8, 9])
for n_clusters in cluster_solutions:
    cluster_labels = hierarchy.cut_tree(Z, n_clusters).flatten()
    sil = silhouette_score(D_avg, cluster_labels, metric="precomputed")
    print("number of clusters = ", n_clusters, 
          "Mean inter-cluster distance = ",
          mean_intercluster_distance(D_avg, cluster_labels, agg="min"), 
          " Average silhouette score = ",
          sil
          )
    sil_scores.append(sil)

# Take solution with highest
n_clusters = cluster_solutions[np.argmax(sil_scores)]
print("the final solution is number of clusters =", n_clusters)

cluster_labels = hierarchy.cut_tree(Z, n_clusters).flatten()

s_vals = silhouette_samples(D_avg, cluster_labels, metric='precomputed')

# Save for later the number of clusters and sorted labels
np.savez(
    "/home/javi/Documentos/meg-excitability-clustering/data/clusters_new.npz", 
          clus_id = hierarchy.cut_tree(Z, n_clusters).flatten(), 
          labels = measures)



