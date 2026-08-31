"""
Script to estimate the region-wise associations between pairs of clusters
"""

# ================================
# 1. Imports
# ================================

import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd

import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import load_data, get_region_labels
from plots import plot_with_overlays

import statsmodels.formula.api as smf
from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats import rankdata

import networkx as nx
from pathlib import Path
from os.path import join as opj
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib import cm


def plot_edge_graph(weight_matrix,
                         measure_labels=None,
                         node_color='white',
                         node_border_color='black',
                         node_fontsize=10,
                         node_padding=8,  # space per character (px²)
                         font_size=10,
                         max_linewidth=5,
                         color="tomato",
                         style="solid",
                         figsize=(8, 6)):

   
    N = weight_matrix.shape[0]
    if measure_labels is None:
        measure_labels = [f"M{i}" for i in range(N)]

    G = nx.Graph()
    G.add_nodes_from(measure_labels)

    all_weights = []

    for i in range(N):
        for j in range(i + 1, N):
            w = weight_matrix[i, j]
            G.add_edge(measure_labels[i], measure_labels[j],
                       weight=w, sign='positive')
            all_weights.append(w)

    # Layout    
    pos = nx.circular_layout(G)
    
    # Compute node sizes dynamically based on label length
    node_sizes = []
    for label in measure_labels:
        size = node_padding + 300  # base size + padding
        node_sizes.append(size)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_color,
                           edgecolors=node_border_color,
                           linewidths=1.5)

    # Draw labels centered (default behavior)
    nx.draw_networkx_labels(G, pos, font_size=node_fontsize, font_weight='bold')

    # Normalize edge widths
    max_weight = max(all_weights) if all_weights else 1

    props = dict(boxstyle='round', facecolor='white', alpha=1)

    # Draw edges with offset for sign
    for u, v, data in G.edges(data=True):
        w = abs(data['weight'])
        width = w / max_weight * max_linewidth
        # Offset direction
        dx = 0.02 
        dy = 0

        x0, y0 = pos[u]
        x1, y1 = pos[v]
        offset_pos = {
            u: (x0 + dx, y0 + dy),
            v: (x1 + dx, y1 + dy)
        }

       
        nx.draw_networkx_edges(G, offset_pos,
                               edgelist=[(u, v)],
                               width=width,
                               edge_color=color,
                               style=style)
        
        if w > 0.0:
            mid_x = (x0 + x1) / 2 + dx
            mid_y = (y0 + y1) / 2 + dy
            plt.text(mid_x, mid_y, f"{int(data['weight'])}%",
                     fontsize=font_size - 15,
                     ha='center', va='center',
                     color="k", bbox=props)
        
    # # Add colorbar
    # sm = plt.cm.ScalarMappable(cmap=color, 
    #                            norm=plt.Normalize(vmin=0, 
    #                                               vmax=weight_matrix.max()))
    # sm.set_array([])  # This is needed for the colorbar to work properly
    # plt.colorbar(sm, label='Node Value')

    #cbar.set_label('Node Value')
    #plt.title(title)
    plt.axis('off')
    #plt.tight_layout()
    plt.show()


# ================================
# 2. Load Data
# ================================

labels = get_region_labels()
data, measures, subjects = load_data()

X = np.array([mat for mat in data.values()])

subjects_df = pd.DataFrame({"Subject":subjects})

# Load clusters
clus_id = np.load("/home/javi/Documentos/meg-excitability-landscape/data/clusters_new.npz")["clus_id"]
measures_clus = np.load("/home/javi/Documentos/meg-excitability-landscape/data/clusters_new.npz")["labels"]

# Make sure label order is the same as in the input data (should be)
assert np.alltrue([a==b for a,b in zip(measures, measures_clus)])

# Convert data into ranks for each subject
X_clus = rankdata(X, axis=2)
del X

X_clus = np.array([np.mean(X_clus[:, clus_id==ii, :], axis=1) for ii in range(max(clus_id)+1)])
X_clus = np.swapaxes(X_clus, 0, 1) # follow the same order as original: subject x measure x region

# Plot distance distributions between pairs of representative measures
subset_measures = [measures_clus[clus_id==ii][0] for ii in range(max(clus_id)+1)]

subset_measures = ["Alpha" if  "Alpha"  in x else x for x in subset_measures]
subset_measures = ["FortyHz" if  "40Hz"  in x else x for x in subset_measures]


# ================================
# 3. statistical analysis
# ================================


# Run region-wise linear mixed models per pairs of clusters
n_regs = 100

zstats, pvals = np.empty((n_regs, len(subset_measures), len(subset_measures))),  \
    np.empty((n_regs, len(subset_measures), len(subset_measures)))
zstats[:] = 0
pvals[:] = 1
for ireg in tqdm(range(n_regs)):
    X_reg = pd.DataFrame(X_clus[:,:,ireg], columns = subset_measures)
    for xx, col_x in enumerate(subset_measures):
        for yy, col_y in enumerate(subset_measures):
            if col_x == col_y:
                continue
            if xx > yy:
                continue
            
            data_df = pd.concat([X_reg.loc[:, [col_x, col_y]], 
                                 subjects_df], axis=1)
            model = smf.mixedlm(f"{col_x}~{col_y}", data=data_df, 
                                groups="Subject")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_fit = model.fit(method='powell', maxiter=1000, reml=False)
            
            if model_fit.converged is False:
                continue
            
            # Calculating explicitly the z-stat
            # This should be equal to the tvalue attribute, but
            # jst in case
            coef = model_fit.fe_params[col_y]
            se = np.sqrt(model_fit.cov_params().loc[col_y, col_y])            
            z = coef/se
            
            zstats[ireg, xx, yy] = z
            zstats[ireg, yy, xx] = zstats[ireg, xx, yy]
            
            pvals[ireg, xx, yy] = model_fit.pvalues[col_y]
            pvals[ireg, yy, xx] = pvals[ireg, xx, yy]
        

zstats = np.nan_to_num(zstats, nan=0)
pvals = np.nan_to_num(pvals, nan=1)

pvals_2d = np.row_stack([squareform(pv, checks=False) for pv in pvals])
zstats_2d = np.row_stack([squareform(z, checks=False) for z in zstats])

pvals_flat = pvals_2d.flatten()

# Correct only valid p-values
_, pvals_2d_corrected, _, _ = multipletests(pvals_flat, method="fdr_bh")

pvals_2d_corrected = pvals_2d_corrected.reshape(pvals_2d.shape)

# Transform back into a 3D format
pvals_corrected  = np.array([squareform(pvals_2d_corrected[ii,:])
                             for ii in range(n_regs)])

# ================================
# 4. plots
# ================================

out_dir = "/home/javi/Documentos/meg-excitability-landscape/plots/across_pairs"
Path(out_dir).mkdir(exist_ok=True, parents=True)

colors = plt.cm.coolwarm(np.linspace(0, 1, 256))
half = len(colors) // 2


blues = plt.cm.Blues(np.linspace(0.3, 1, 128))  # darker blue
reds = plt.cm.Reds(np.linspace(0.3, 1, 128))      # darker red
# Lower half colormap
lower_half = plt.cm.colors.ListedColormap(blues)

# Upper half colormap
upper_half = plt.cm.colors.ListedColormap(reds)

# Plot individual brain plots with significant results
for ii in range(6):
    for jj in range(ii+1, 6):
        for sign in ["positive", "negative"]:
            conn_dict = dict()
            overlay_dict = dict() 
            outline_dict = dict()
            for kk in range(n_regs):
                z = zstats[kk, ii, jj]
                pv = pvals_corrected[kk, ii, jj]
                if sign=="positive":
                    mask = z>0
                    cmap=upper_half#"OrRd"
                    cmap="OrRd"
                    vmax = np.nanmax(zstats)
                    vmin = 0
                else:
                    mask = z<0
                    #cmap= lower_half#"GnBu"
                    cmap= "GnBu"
                    vmax = 0
                    vmin = np.nanmin(zstats)
                    
                conn_dict[labels[kk]] = z*(mask)*(pv<0.05)
                overlay_dict[labels[kk]] = z*(mask)
                if  mask*(pv<0.05):
                    outline_dict[labels[kk]] = 1 + 1e-16*np.random.randn() 
                else:
                    outline_dict[labels[kk]] = 0
            
            # highlighted
            fig = plot_with_overlays(overlay_dict, 
                                        cbar=True, 
                                        cmap=cmap, 
                                        outline_dict=outline_dict, 
                                        conn_overlay_dict=conn_dict,
                                        vmax=vmax,
                                        vmin=vmin,
                                        alpha=0.4)
    
            #fig.axes[0].set_title(strategy, size=20)
            fig.axes[1].set_title("Z-stat", size=15)
            #fig.axes[1].set_xticks([fig.axes[1].get_xticks()[0], 
             #                       fig.axes[1].get_xticks()[-1]])
            #fig.axes[1].set_xticklabels(["Less", "More"])
            fig.axes[1].tick_params(labelsize=10)            
            #fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
            
            
            # Add title:                
            label_x = subset_measures[ii]
            label_y = subset_measures[jj]
            
            if label_x == subset_measures[0]:
                label_x = "40Hz"
            if label_y == subset_measures[0]:
                label_x = "40Hz"
            fig.axes[0].set_title(f"{label_x} vs {label_y}", pad=-3, size=25)
            
            fig.tight_layout()
            
            Path(out_dir + "/" +  
                 f"{subset_measures[ii]}_{subset_measures[jj]}").mkdir(exist_ok=True, parents=True)
            
            fig.savefig(opj(out_dir, f"{subset_measures[ii]}_{subset_measures[jj]}",
                            f"{subset_measures[ii]}_{subset_measures[jj]}_{sign}_highlighted.png"), 
                        dpi=300)
            fig.savefig(opj(out_dir, f"{subset_measures[ii]}_{subset_measures[jj]}",
                            f"{subset_measures[ii]}_{subset_measures[jj]}_{sign}_highlighted.svg"), 
                        dpi=300)
            plt.close(fig)
            
    

mask = np.isnan(zstats_2d)


## Plot Heatmapt with z-stats (Reg x pairs)

# Get the base colormaps
pu_bu = cm.get_cmap('PuBu_r', 128)   # 128 colors from PuBu
or_rd = cm.get_cmap('OrRd', 128)   # 128 colors from OrRd

# Combine them
combined_colors = np.vstack((pu_bu(np.linspace(0, 1, 128)),
                             or_rd(np.linspace(0, 1, 128))))
combined_cmap = ListedColormap(combined_colors)

for sign in ["positive", "negative"]:
    fig, ax  = plt.subplots(figsize=(10,5))
    
    sns.heatmap(zstats_2d.T, cmap=combined_cmap, center=0, 
                vmin=np.nanmin(zstats_2d), 
                vmax=np.nanmax(zstats_2d), 
                linewidth=0., 
                linecolor='k', cbar=False, xticklabels=False, 
                yticklabels=False, mask=mask.T, ax=ax)

    # Add X markers on masked cells
    for i in range(zstats_2d.T.shape[0]):
        for j in range(zstats_2d.T.shape[1]):
            if mask.T[i, j]:
                # Coordinates: (x_start, y_start) to (x_end, y_end)
                x_start, y_start = j, i
                x_end, y_end = j + 1, i + 1
                ax.plot([x_start, x_end], [y_start, y_end], color='k', 
                         linewidth=1)
                ax.plot([x_start, x_end], [y_end, y_start], color='k', 
                         linewidth=1)
    # Add X markers on masked cells
    for i in range(zstats_2d.T.shape[0]):
        for j in range(zstats_2d.T.shape[1]):
            if pvals_2d_corrected.T[i, j]<0.05:
               rect = patches.Rectangle((j, i), 1, 1,
                                         linewidth=1.5,
                                         edgecolor='k',
                                         facecolor='none')
               if sign == "positive":
                   if zstats_2d.T[i,j]>0:
                       ax.add_patch(rect)
               else:
                   if zstats_2d.T[i,j]<0:
                       ax.add_patch(rect)    
    ax.set_ylabel("Pairs", size=25)
    ax.set_xlabel("Region", size=25)
    # Make the main axes (spines) thicker
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)  # adjust thickness as needed
        spine.set_edgecolor("black")  # or another color you prefer
        
    outfile = "/home/javi/Documentos/meg-excitability-clustering/plots/"+\
        f"heatmap_zstats_{sign}_clusters.png"         
    
        
    plt.savefig(outfile, dpi=300)
    plt.savefig(outfile.replace(".png", ".svg"), dpi=300)
