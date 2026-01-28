"""
Contains functions for plotting
"""

from neuromaps.datasets import fetch_fslr
from surfplot import Plot
import numpy as np
import matplotlib.pylab as plt


def plot_parcellated_data(conn_dict, cbar=True, cmap='viridis', 
                          outline_dict = None, zero_transparent=True, 
                          conn_overlay_dict = None, alpha=1, color_range=None):
    from neuromaps.images import dlabel_to_gifti
    
    with open(
            "/home/javi/Documentos/meg-excitability-clustering/data/Schaefer2018_100Parcels_17Networks_order_info.txt") as f:
        lines = f.readlines()
           
    labels_schaefer = dict()
    for ii in range(int(len(lines)/2)):
        name = lines[int(2*ii)].strip()
        index = int(lines[int(2*ii)+1].split(" ")[0])
        labels_schaefer[index] = name 
    
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh)
    # add schaefer parcellation (no color bar needed)
    lh_parc, rh_parc = dlabel_to_gifti(
        "/home/javi/Documentos/meg-excitability-clustering/data/Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
        )
    lh_parc = lh_parc.agg_data()
    rh_parc = rh_parc.agg_data()
    #load_parcellation('schaefer', scale=100)
    lh_parc_con  = np.zeros_like(lh_parc, dtype=float)
    rh_parc_con = np.zeros_like(rh_parc, dtype=float)
    
    for index, name in labels_schaefer.items():
        lh_parc_con[lh_parc==index] = conn_dict[name]
    
    for index, name in labels_schaefer.items():
        rh_parc_con[rh_parc==index] = conn_dict[name]
    
    if conn_overlay_dict:
        lh_overlay  = np.zeros_like(lh_parc, dtype=float)
        rh_overlay = np.zeros_like(rh_parc, dtype=float)
        
        for index, name in labels_schaefer.items():
            lh_overlay[lh_parc==index] = conn_overlay_dict[name]
        
        for index, name in labels_schaefer.items():
            rh_overlay[rh_parc==index] = conn_overlay_dict[name]
        
        p.add_layer({'left': lh_overlay, 'right': rh_overlay}, 
                    cbar=False, cmap=cmap, 
                    zero_transparent=zero_transparent, alpha=alpha, color_range=color_range)
        
    p.add_layer({'left': lh_parc_con, 'right': rh_parc_con}, 
                cbar=cbar, cmap=cmap, 
                zero_transparent=zero_transparent, color_range=color_range)
    
    if type(outline_dict) == dict:
        lh_outline  = np.zeros_like(lh_parc, dtype=float)
        rh_outline = np.zeros_like(rh_parc, dtype=float)
        
        for index, name in labels_schaefer.items():
            if name in outline_dict.keys():
                lh_outline[lh_parc==index] = outline_dict[name]
        
        for index, name in labels_schaefer.items():
            if name in outline_dict.keys():
                rh_outline[rh_parc==index] = outline_dict[name]
        
        p.add_layer({'left': lh_outline, 'right': rh_outline}, 
                    cmap="gray", as_outline=True, cbar=False)
                    
    
    fig = p.build()
    return fig

def plot_with_overlays(conn_dict, 
                       cbar=True, 
                       cmap='viridis', 
                       outline_dict = None, 
                       zero_transparent=True, 
                       conn_overlay_dict = None, 
                       alpha=1, 
                       vmax = None,
                       vmin = None
                       ):
    from neuromaps.images import dlabel_to_gifti
    from neuromaps.datasets import fetch_fslr
    from surfplot import Plot
    import numpy as np
    
    with open(
            "/home/javi/Documentos/meg-excitability-clustering/data/Schaefer2018_100Parcels_17Networks_order_info.txt") as f:
        lines = f.readlines()
           
    labels_schaefer = dict()
    for ii in range(int(len(lines)/2)):
        name = lines[int(2*ii)].strip()
        index = int(lines[int(2*ii)+1].split(" ")[0])
        labels_schaefer[index] = name 
    
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh)
    # add schaefer parcellation (no color bar needed)
    lh_parc, rh_parc = dlabel_to_gifti(
        "/home/javi/Documentos/meg-excitability-clustering/data/Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
        )
    lh_parc = lh_parc.agg_data()
    rh_parc = rh_parc.agg_data()
    #load_parcellation('schaefer', scale=100)
    lh_parc_con  = np.zeros_like(lh_parc, dtype=float)
    rh_parc_con = np.zeros_like(rh_parc, dtype=float)
    
    for index, name in labels_schaefer.items():
        lh_parc_con[lh_parc==index] = conn_dict[name]
    
    for index, name in labels_schaefer.items():
        rh_parc_con[rh_parc==index] = conn_dict[name]
    
    
    if conn_overlay_dict:
        lh_overlay  = np.zeros_like(lh_parc, dtype=float)
        rh_overlay = np.zeros_like(rh_parc, dtype=float)
        
        if vmax is None:
            vmax = max(conn_overlay_dict.values())
        if vmax is None:
            vmin = min(conn_overlay_dict.values())
        color_range = (vmin, vmax)        
        for index, name in labels_schaefer.items():
            lh_overlay[lh_parc==index] = conn_overlay_dict[name]
        
        for index, name in labels_schaefer.items():
            rh_overlay[rh_parc==index] = conn_overlay_dict[name]
        
        p.add_layer({'left': lh_overlay, 'right': rh_overlay}, 
                    cbar=True, cmap=cmap, color_range=color_range,
                    zero_transparent=zero_transparent)
    
        p.add_layer({'left': lh_parc_con, 'right': rh_parc_con}, 
                    cbar=False, cmap=cmap,  alpha=alpha, color_range=color_range,
                    zero_transparent=zero_transparent)
    else:
        p.add_layer({'left': lh_parc_con, 'right': rh_parc_con}, 
                    cbar=cbar, cmap=cmap,  
                    zero_transparent=zero_transparent)
        
    if type(outline_dict) == dict:
        lh_outline  = np.zeros_like(lh_parc, dtype=float)
        rh_outline = np.zeros_like(rh_parc, dtype=float)
        
        for index, name in labels_schaefer.items():
            if name in outline_dict.keys():
                lh_outline[lh_parc==index] = outline_dict[name]
        
        for index, name in labels_schaefer.items():
            if name in outline_dict.keys():
                rh_outline[rh_parc==index] = outline_dict[name]
        
        p.add_layer({'left': lh_outline, 'right': rh_outline}, 
                    cmap="gray_r", as_outline=True, cbar=False)
                    
    
    cbar_kws = dict(outer_labels_only=False, pad=.02, n_ticks=3, shrink=2)
    fig = p.build(cbar_kws=cbar_kws)
#    fig = p.build()
    return fig

