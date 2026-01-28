"""
Conduct the separability analysis, based on region-wise ANOVA 
applied to the centroids of the clusters.

"""

# ================================
# 1. Imports
# ================================

import numpy as np
from scipy.stats import rankdata
from scipy.stats import f_oneway
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-clustering/src")
from input_data import get_region_labels, load_data
from plots import plot_parcellated_data, circular_lollipop_filled
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt

# ================================
# 2. Load Data
# ================================

# ----- PSD data -----

labels = get_region_labels()
psd_mats, strategies = load_data()

# stack matrices
X = np.array(list(psd_mats.values()))

X_ranks = np.array([rankdata(X[ii,:,:], axis=1) for ii in range(X.shape[0])])

# Now, compute centroids by averaging per cluster
clus_id = np.load(
    "/home/javi/Documentos/meg-excitability-clustering/data/clusters_new.npz")["clus_id"]

X_avg_clus = np.array([X_ranks[:, clus_id==label,:].mean(axis=1) 
                       for label in np.unique(clus_id)])
X_avg_clus = np.swapaxes(X_avg_clus, 0, 1)

# ================================
# 3. Calculate separability using ANOVA
# ================================

n_regs = X_avg_clus.shape[-1]
assert n_regs == 100

n_clus = X_avg_clus.shape[-2]
assert n_clus == 6

feature_importances = dict()
for jj in range(n_regs):
    feature_importances[labels[jj]] = f_oneway(
        *[X_avg_clus[:,ii,jj] for ii in range(n_clus)]).statistic
    
# ================================
# 4. Plot -- Regions
# ================================

# Region importance
fig = plot_parcellated_data(feature_importances, cbar=True, cmap="YlOrRd")
#fig.axes[0].set_title(strategy, size=20)
fig.axes[1].set_title("Importance", size=15)
fig.axes[1].set_xticks([fig.axes[1].get_xticks()[0], 
                        fig.axes[1].get_xticks()[-1]])
fig.axes[1].set_xticklabels(["Less", "More"])
fig.axes[1].tick_params(labelsize=10)
fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
fig.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/region_importance.png", dpi=300)
fig.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/region_importance.svg", dpi=300)


# Region importance with 4 highlighted regions
outline_dict = {}
ix_highlight = [5, 24, 47, 96]
#ix_highlight = [96]
for key in np.array(sorted(feature_importances, key=feature_importances.get, 
                           reverse=True))[ix_highlight]:
    outline_dict[key] = feature_importances[key]

fig = plot_parcellated_data(feature_importances, cbar=False, cmap="YlOrRd", 
                            outline_dict=outline_dict)
fig.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/region_importance_highlight.png", 
            dpi=300)
fig.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/region_importance_highlight.svg", 
            dpi=300)

# Barplots for each of these highlighted regions
colors = ["C3", "C1", "C2", "C4", "C5", "C6"]
for key in outline_dict.keys():
    ix = np.where(np.array(labels) == key)[0]
    to_plot = pd.DataFrame(np.squeeze(X_avg_clus[:,:,ix]), 
                           columns = ["40Hz", "Alpha", "DFA", 
                                      "Exponent", "Phase_Ex","SE"])
    to_plot_melt = pd.melt(to_plot)
    

    fig, ax = plt.subplots()
    sns.violinplot(to_plot_melt, x="variable", y="value", 
                   inner=None,
                    palette="pastel",edgecolor="black", 
                    density_norm="area")
    

    for color, collection in zip(colors, ax.collections):
         #collection.set_facecolor(color)
         collection.set_alpha(0.5)
         collection.set_facecolor(color)
         collection.set_edgecolor(color)
         collection.set_linewidth(1)
    
    
    sns.pointplot(x="variable", y="value", data = to_plot_melt,
                  linestyle="none", errorbar=None,
                  marker="_", markersize=10, markeredgewidth=3, color="k")
    
    sns.stripplot(to_plot_melt, x="variable", y="value",  
                  jitter=0.05, size=2, 
                  color="k",  zorder=5)
    
    for color, collection in zip(colors, ax.collections[6:]):
         #collection.set_facecolor(color)
         collection.set_alpha(0.5)

         collection.set_facecolor(color)         
         collection.set_edgecolor(color)
         #patch.set_edgecolor(color)
         collection.set_linewidth(0.5)
    ax.set_xlabel("")
    ax.set_ylabel("E/I values (Ranked)", size=22)
    #ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=10, size=0)
    ax.set_title("_".join(key.split("_")[1:]), size=28)
    
    plt.savefig(f"/home/javi/Documentos/meg-excitability-clustering/plots/region_anova_{key}_v2.png", 
                dpi=300)
    plt.savefig(f"/home/javi/Documentos/meg-excitability-clustering/plots/region_anova_{key}_v2.svg", 
                dpi=300)
    
    
# Plot legend separately
colors = ["C3", "C1", "C2", "C4", "C5", "C6"]

from matplotlib.patches import Patch

legend_elements = []
for color, label in zip(colors, ["40Hz", "Alpha", "DFA", 
                                 "Aperiodic", "EI","SE"]):
    legend_elements.append(Patch(facecolor=color, label=label))

fig, ax = plt.subplots()
ax.axis('off')
ax.legend(handles=legend_elements, ncols= 3, 
          loc='center', frameon=False, fontsize=15)
plt.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/legend_clusters.png", 
            bbox_inches='tight', dpi=300)
plt.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/legend_clusters.svg", 
            bbox_inches='tight', dpi=300)


# ================================
# 5. Plot -- Networks
# ================================
   
# Plot aggregagating improtance across major resting state networks
labels_rsn = [label.split("_")[2] for label in labels]
feat_imp_rsn = dict()
for label in np.unique(labels_rsn):
    # take those regions beloging to this network
    feat_imp_rsn[label] = np.mean([value for key, value in 
                                   feature_importances.items() if label in key])


fig, ax = circular_lollipop_filled(feat_imp_rsn)
fig.tight_layout()
fig.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/rsn_importance.png", dpi=300)
fig.savefig("/home/javi/Documentos/meg-excitability-clustering/plots/rsn_importance.svg", dpi=300)

# Plot each region of the RSN network
for rsn_label in np.unique(labels_rsn):
    roiplots_dict = {}
    for label in labels:
        if rsn_label in label:
            roiplots_dict[label] = 1
        else:
            roiplots_dict[label] = 0
    fig = plot_parcellated_data(roiplots_dict, cbar=False, cmap="binary", 
                                conn_overlay_dict=roiplots_dict)
    fig.savefig(f"/home/javi/Documentos/meg-excitability-clustering/plots/{rsn_label}.png", dpi=300)
    fig.savefig(f"/home/javi/Documentos/meg-excitability-clustering/plots/{rsn_label}.svg", dpi=300)
    