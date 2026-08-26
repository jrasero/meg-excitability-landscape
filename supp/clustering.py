#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:51:08 2026

@author: javi
"""

from glob import glob
from os.path import join as opj
from scipy.io import loadmat
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm
import mat73
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import get_region_labels
from plots import plot_parcellated_data
from pathlib import Path
data_folder = "/home/javi/Documentos/meg-excitability-landscape/supp/Excitability_200Parcels/data"
measures = [ meas.split("/")[-1] for meas in glob(opj(data_folder, "*"))]

data_dict = dict()
for measure in tqdm(measures):

    data_files = glob(opj(data_folder, measure,"**", "*.mat"), recursive=True)

    try:
        loadmat(data_files[0])
        load_data = loadmat
    except NotImplementedError:
        load_data = mat73.loadmat

    if measure in ["40Hz", "40Hz_Zscore"]:
        
        data = [np.squeeze(load_data(d)["ITPC_export"]["values"][0,0]) for d in data_files]
    else:
        data = [np.squeeze(load_data(d)["data"])[2:] for d in data_files]
    
    if measure in ["FOOOF_Alpha", "AbsoluteAlpha", "AlphaRelative", 
                   "SE", "Exponent", "Offset"]:
        
        data = [d.max() -  d + d.min() for d in data]
        
    data = [rankdata(d) for d in data]
        
    data_dict[measure] = np.array(data)
        
#===========================
# 2 - Plot each group average
#===========================
labels = get_region_labels(n_rois=200)
Path("/home/javi/Documentos/meg-excitability-landscape/supp/plots_schaefer200").mkdir(parents=True, exist_ok=True)
for ii, measure in enumerate(data_dict.keys()):
    
    psd_avg_data = data_dict[measure].mean(axis=0).copy()
    psd_profiles_dict = {}
    for key, value in zip(labels, psd_avg_data):
        psd_profiles_dict[key] = value
    
    title=measure
    # This is just to change the title for the names of these two 
    # particular measures
    if measure=="DFA_EI":
        title = "DFA"
    elif measure=="Phase_Ex":
        title = "EI"
    fig = plot_parcellated_data(psd_profiles_dict, n_rois=200)
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


R_subjects = []
for ii in range(72):
    data_meas = []
    for key,value in data_dict.items():
        data_meas.append(value[ii,:])
        
    R_subjects.append(pd.DataFrame(data_meas).T.corr("spearman").to_numpy())

R_subjects = np.array(R_subjects)
R_avg = np.array(R_subjects).mean(0)

D_subjects = 1 - R_subjects # Distance matrix by subject
D_avg = 1 - R_avg # Distance matrix for the group


strategies = list(data_dict.keys())
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
                          labels=strategies, 
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


# And the actual correlation matrix below
im = axs[1].imshow(R_avg[dn["leaves"],:][:, dn["leaves"]], aspect="auto")

square_positions = [(0, 0), (4, 4), (5, 5), (6, 6)]
square_size = [4, 1, 1, 2]
square_colors = ["C3", "C2", "C5", "C4", "C1"]

for (x,y), sq, c in zip(square_positions, square_size, square_colors):
    rect = patches.Rectangle(
            (x-0.5, y-0.5), sq, sq,
            linewidth=5, edgecolor=c, facecolor='none', linestyle="--",
            zorder=20,  # ensures it's on top

        )
    axs[1].add_patch(rect)

axs[1].set_xticks(np.arange(len(strategies)))
axs[1].set_xticklabels(ticklabels, rotation=45)
axs[1].tick_params(labelsize=20, axis="x")
axs[1].tick_params(labelsize=0, axis="y")
plt.tight_layout()