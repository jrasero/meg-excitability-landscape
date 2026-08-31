"""
Contains functions for plotting
"""

from neuromaps.datasets import fetch_fslr
from surfplot import Plot
import numpy as np
import matplotlib.pylab as plt


def plot_parcellated_data(conn_dict, cbar=True, cmap='viridis', 
                          outline_dict = None, zero_transparent=True, 
                          conn_overlay_dict = None, alpha=1, color_range=None, n_rois=100):
    from neuromaps.images import dlabel_to_gifti
    
    with open(
            f"/home/javi/Documentos/meg-excitability-landscape/data/Schaefer2018_{n_rois}Parcels_17Networks_order_info.txt") as f:
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
        f"/home/javi/Documentos/meg-excitability-landscape/data/Schaefer2018_{n_rois}Parcels_17Networks_order.dlabel.nii"
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
            "/home/javi/Documentos/meg-excitability-landscape/data/Schaefer2018_100Parcels_17Networks_order_info.txt") as f:
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
        "/home/javi/Documentos/meg-excitability-landscape/data/Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
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

def circular_lollipop_filled(values_dict, title='',
                              stick_color='k', dot_color='k', 
                              fill_color='#A0A0A0', alpha=1, figsize=(10, 10)):
    """
    Circular lollipop plot with sticks, joined dots, and filled area under the curve.

    Parameters:
        labels (list of str): Categories.
        values (list of float): Corresponding values.
        title (str): Title of the plot.
        stick_color (str): Color of the stick lines.
        dot_color (str): Color of the dots.
        fill_color (str): Color of the area fill under the curve.
        alpha (float): Transparency of the fill.
        figsize (tuple): Figure size.
    """
    
    # Compute angles for each lollipop
    labels = list(values_dict.keys())
    values = list(values_dict.values())
    
    if len(labels) != len(values):
        raise ValueError("Length of labels and values must match.")

    # Setup
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Repeat first value to close the circle for fill and line
    angles += angles[:1]
    values += values[:1]
    labels += labels[:1]

    # Plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Draw lollipop sticks
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.plot([angle, angle], [0, value], color=stick_color, linewidth=0.)

    # Draw filled area under the curve
    ax.fill(angles, values, color=fill_color, alpha=alpha)

    # Connect the dots with lines
    ax.plot(angles, values, color=dot_color, linewidth=2)

    # Draw dots
    ax.scatter(angles, values, color=dot_color, s=120, zorder=3)

    # Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], size=21)
    ax.set_yticklabels([])
    ax.set_ylim(0, max(values) * 1.1)
    ax.spines['polar'].set_visible(False)
    ax.grid(color='k', linestyle='dotted', linewidth=0.5)

    plt.title(title, size=16)
    return fig, ax
