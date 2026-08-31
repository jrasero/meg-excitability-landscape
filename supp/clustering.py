#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:51:08 2026

@author: javi
"""
import numpy as np
from scipy.stats import rankdata
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import get_region_labels
from plots import plot_parcellated_data
from pathlib import Path

# Load data        
data_mats = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_schaefer200.npz")["data"]
measures =  np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_schaefer200.npz")["measures"]

#===========================
# 2 - Plot each group average
#===========================
labels = get_region_labels(n_rois=200)
Path("/home/javi/Documentos/meg-excitability-landscape/supp/plots_schaefer200").mkdir(parents=True, exist_ok=True)
for mix, measure in enumerate(measures):
    
    # Rank region of each scan session
    data_avg_data = rankdata(data_mats[:,mix,:], axis=1).mean(axis=0).copy()
    data_profiles_dict = {}
    for key, value in zip(labels, data_avg_data):
        data_profiles_dict[key] = value
    
    title=measure
    # This is just to change the title for the names of these two 
    # particular measures
    if measure=="DFA_EI":
        title = "DFA"
    elif measure=="Phase_Ex":
        title = "EI"
    fig = plot_parcellated_data(data_profiles_dict, n_rois=200)
    # fig.axes[0].set_title(title, size=20)
    # fig.axes[1].tick_params(labelsize=12)
    # fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
    # ticks = fig.axes[1].get_xticks()
    # fig.axes[1].set_xticks([ticks[0], ticks[-1]])
    # fig.axes[1].set_xticklabels(["min", "max"])
    fig.axes[1].remove()
    fig.savefig(
        f"/home/javi/Documentos/meg-excitability-landscape/supp/plots_schaefer200/brainplot_{measure}.png", 
                dpi=300, bbox_inches="tight")


# ================================
# 3. Clustering
# ================================

R_scans = []
for six in range(72):
    R_scans.append(pd.DataFrame(data_mats[six, :,:]).T.corr("spearman").to_numpy())

R_scans = np.array(R_scans)
R_avg = np.array(R_scans).mean(0)

D_subjects = 1 - R_scans # Distance matrix by subject
D_avg = 1 - R_avg # Distance matrix for the group

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

#outdir= "/home/javi/Documentos/projects/Epilepsy_TLE_MEG_Excitability/plots"
fig, ax = plt.subplots(figsize=(18,10))

# Plot dendogram
dn = hierarchy.dendrogram(Z, 
                          labels=measures, 
                          color_threshold =0.,
                          #link_color_func=lambda x: link_cols[x],
                          above_threshold_color="xkcd:dark grey",
                          ax=ax)

ticklabels = dn["ivl" ] 

# Rename some of the measures
for ii, name in enumerate(ticklabels):
    if name == "DFA_EI":
        ticklabels[ii] = "DFA"
    elif name == "Phase_Ex":
        ticklabels[ii] = "EI"
      
ax.set_xticklabels(ticklabels)
for collection in ax.collections:
    collection.set_linewidth(2)  # Set to your desired width
    
ax.tick_params(axis="y", labelsize=0, size=0, length=0, width=0, colors="w")
# Increase the line width
for collection in ax.collections:
    collection.set_linewidth(2)  # Set to your desired width

fig.savefig("/home/javi/Documentos/meg-excitability-landscape/supp/plots_schaefer200/dendogram_clustering.svg")
fig.savefig("/home/javi/Documentos/meg-excitability-landscape/supp/plots_schaefer200/dendogram_clustering.png")
