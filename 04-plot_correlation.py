"""
Plot spatial correlation between measures (Fig2B)

"""

# ================================
# 1. Imports + Function to plot
# ================================

import numpy as np
import pandas as pd
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-clustering/src")
from input_data import load_data, get_region_labels

def plot_edge_graph(weight_matrix,
                         measure_labels=None,
                         node_color='white',
                         node_border_color='black',
                         node_fontsize=10,
                         node_padding=8,  # space per character (px²)
                         font_size=10,
                         max_linewidth=5,
                         cmap="viridis",
                         style="solid",
                         threshold=0,
                         figsize=(8, 6)):
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.cm as matplotlib_cmap

    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch


    cm = matplotlib_cmap.get_cmap(cmap)
    N = weight_matrix.shape[0]
    if measure_labels is None:
        measure_labels = [f"M{i}" for i in range(N)]

    G = nx.Graph()
    G.add_nodes_from(measure_labels)

    all_weights = []

    for i in range(N):
        for j in range(i + 1, N):
            w = weight_matrix[i, j]
            if w > threshold:
                G.add_edge(measure_labels[i], measure_labels[j],
                           weight=w)
                all_weights.append(w)

    # Layout    
    pos = nx.circular_layout(G)
    
    # Normalize for color mapping
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    
    norm = Normalize(vmin=-max(abs(weights)), vmax=max(abs(weights)))
    colors = [cm(norm(w)) for w in weights]
    widths = 2 + 10 * (weights - weights.min()) / (weights.max() - weights.min())
    min_alpha = 0.3
    max_alpha = 1.0

    alphas = min_alpha + (max_alpha - min_alpha) * norm(weights)
    # Compute node sizes dynamically based on label length
    node_sizes = []
    for label in measure_labels:
        size = node_padding + 300  # base size + padding
        node_sizes.append(size)

    # Draw nodes
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_color,
                           edgecolors=node_border_color,
                           linewidths=2.5, ax=ax)

    # Draw labels centered (default behavior)
    nx.draw_networkx_labels(G, pos, font_size=node_fontsize, 
                            font_weight='bold', ax=ax)

    # Normalize edge widths
#    max_weight = max(all_weights) if all_weights else 1

 #   props = dict(boxstyle='round', facecolor='white', alpha=1)
    
    # Draw curved edges manually
    for ((u, v), color, width, alpha) in zip(G.edges(), colors, widths, alphas):
        src = pos[u]
        dst = pos[v]
    
        arrow = FancyArrowPatch(
            src, dst,
            connectionstyle="arc3,rad=0.1",
            color=color,
            linewidth=width,
            arrowstyle='-',
            alpha=alpha
        )
        ax.add_patch(arrow)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
#    cbar = plt.colorbar(sm, ax=ax, label='Correlation Strength')


    #cbar.set_label('Node Value')
    #plt.title(title)
    plt.axis('off')
    #plt.tight_layout()
    return fig, ax, sm
 
    
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

# Load clusters
res_clus = np.load(
    "/home/javi/Documentos/meg-excitability-landscape/data/clusters_new.npz")

R_avg_ordered = R_avg.copy()[np.argsort(res_clus["clus_id"]),:][:, np.argsort(res_clus["clus_id"])]
measure_labels = res_clus["labels"][np.argsort(res_clus["clus_id"])]
fig, ax, sm = plot_edge_graph(R_avg_ordered, 
                              measure_labels=measure_labels,
                              node_fontsize=10,
                              cmap="RdBu_r",
                              font_size=25, threshold=-np.inf,
                              node_padding=7000,
                              figsize=(14,12))
# Now customize the colorbar
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', 
                    fraction=0.032, pad=0.04)

# Custom styling
cbar.set_label('Correlation strength', fontsize=25)
cbar.ax.tick_params(labelsize=20)
ax.margins(0.1)  

fig.savefig("/home/javi/Documentos/meg-excitability-landscape/plots/correlation_measures_graph_v2.png", 
            dpi=300)
fig.savefig("/home/javi/Documentos/meg-excitability-landscape/plots/correlation_measures_graph_v2.svg", 
            dpi=300)