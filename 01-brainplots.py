"""
This script creates the brainplots for 
each EI measurements.

"""

# ================================
# 1. Imports
# ================================

import numpy as np
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import load_data, get_region_labels
from plots import plot_parcellated_data


# ================================
# 2. Load Data
# ================================

labels = get_region_labels()
data, measures, _ = load_data()
data = np.array(list(data.values()))
avg_data = data.mean(axis=0) # Group average

# ================================
# 3. Plot
# ================================

for ii, meas in enumerate(measures):
    
    data_profiles_dict = {}
    for key, value in zip(labels, avg_data[ii,:]):
        data_profiles_dict[key] = value
    
    title=meas
    # This is just to change the title for the names of these two 
    # particular measures
    if meas=="DFA_EI":
        title = "DFA"
    elif meas=="Phase_Ex":
        title = "EI"
    fig = plot_parcellated_data(data_profiles_dict)
    fig.axes[0].set_title(title, size=20)
    fig.axes[1].tick_params(labelsize=12)
    fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
    ticks = fig.axes[1].get_xticks()
    fig.axes[1].set_xticks([ticks[0], ticks[-1]])
    fig.axes[1].set_xticklabels(["min", "max"])
    fig.savefig(
        f"/home/javi/Documentos/meg-excitability-landascape/plots/brainplot_{meas}.png", 
                dpi=300, bbox_inches="tight")
