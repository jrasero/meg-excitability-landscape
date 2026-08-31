import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import get_region_labels
from plots import plot_parcellated_data


data_mats = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_timewindows.npz")["data"]
measures = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_timewindows.npz")["measures"]

n_windows = data_mats.shape[-1]
n_scans = data_mats.shape[0]
labels = get_region_labels()

Path("/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters").mkdir(parents=True, exist_ok=True)
for mix, measure in enumerate(measures):
    
    if measure == "Phase_Ex":
        measure="EI"
    elif measure == "DFA_EI":
        measure = "DFA"
    elif measure == "AbsoluteAlpha":
        measure = "Alpha"
    
    avg_data = data_mats[:,mix, :, :].copy().mean(axis=0) #data_dict[measure].copy().mean(axis=0)
    
    for w_id in range(n_windows):
        Path(f"/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters/window_{w_id+1}").mkdir(parents=True, exist_ok=True)

        data_profiles_dict = {}
        for key, value in zip(labels, avg_data[:, w_id]):
            data_profiles_dict[key] = value
        
        title=measure
        # This is just to change the title for the names of these two 
        # particular measures
        if measure == "Phase_Ex":
            title="EI"
        elif measure == "DFA_EI":
            title = "DFA"
        elif measure == "AbsoluteAlpha":
            title = "Alpha"
        fig = plot_parcellated_data(data_profiles_dict)
        
        fig.axes[1].remove()
        fig.savefig(
            f"/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters/window_{w_id+1}/brainplot_{measure}.png",
                    dpi=300, bbox_inches="tight")


between_measures_corrs = []
for w_id in range(n_windows): 
    for ii, mx in enumerate(measures):
        for jj, my in enumerate(measures):
            if mx == my:
                continue
            if ii>jj:
                continue
            for sid in range(n_scans):
    
                between_measures_corrs.append(
                    [w_id, 
                     f"{mx}-{my}", 
                     sid, 
                     spearmanr(data_mats[sid, ii, :, w_id], 
                               data_mats[sid, jj, :, w_id])[0]
                     ])
        
    
between_measures_corrs = pd.DataFrame(between_measures_corrs, 
                                      columns=["window", "pair", "subject", "rho"])    

between_measures_corrs_melt = pd.melt(between_measures_corrs, 
                                      id_vars=["window", "pair", "rho"])
fig, ax = plt.subplots(figsize=(15,10))
sns.pointplot(x = "window", y="rho", hue="pair", data=between_measures_corrs_melt, ax=ax)#, 
              #capsize=.05)
ax.set_ylabel(r'$\rho$', size=25)
ax.set_xticklabels(["T" + str(ii+1) for ii in range(n_windows)])
ax.set_xlabel("")
ax.tick_params(labelsize=20)
leg = ax.get_legend()
handles = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles
labels = [t.get_text() for t in leg.get_texts()]

leg.remove()

ax.legend(
    handles,
    labels,
    ncol=3,
    loc=8,#(0.3, 0.01),
    edgecolor="k",
    fontsize=12
)

fig.savefig("/home/javi/Documentos/meg-excitability-landscape/supp/plots/corrs_windows.svg", dpi=300)
fig.savefig("/home/javi/Documentos/meg-excitability-landscape/supp/plots/corrs_windows.png", dpi=300)