from glob import glob
import seaborn as sns
from os.path import join as opj
from scipy.io import loadmat
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import mat73
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from input_data import get_region_labels, load_data
from plots import plot_parcellated_data


def reverse_vals(d):
    # Make sure I am reversing the right axis when using with the time windows
    if len(d) != 100:
        raise ValueError
    return d.max() -  d + d.min()
    

data_folder = "/home/javi/Documentos/meg-excitability-landscape/supp/Excitability_TimeResolved"
measures = [ meas.split("/")[-1] for meas in glob(opj(data_folder, "*"))]

# Only clusters
measures  = ['Phase_Ex','DFA_EI', 'AbsoluteAlpha', 'Exponent', 'SE']

data_dict = dict()
for measure in tqdm(measures):

    data_files = glob(opj(data_folder, measure,"**", "*.mat"), recursive=True)
    
    # Discard "NP819", as it is incmplete
    data_files = [filename for filename in data_files if "NP819" not in filename]
    print(len(data_files))
    try:
        loadmat(data_files[0])
        load_data = loadmat
    except NotImplementedError:
        load_data = mat73.loadmat

    if measure == "FOOOF_Alpha":
        data = [load_data(d)["fooof_alpha_power_data"][2:,:] for d in data_files]
    elif measure == "AlphaRelative":
        data = [load_data(d)["rel_alpha_power_data"][2:,:] for d in data_files]
    else:
        data = [load_data(d)["data"][2:,:] for d in data_files]
    
    if measure in ["FOOOF_Alpha", "AbsoluteAlpha", "AlphaRelative", 
                   "SE", "Exponent", "Offset"]:
        
        data = [np.apply_along_axis(reverse_vals, axis=0, arr=d) for d in data]
        #data = [d.max() -  d + d.min() for d in data]
        
    #data = [rankdata(d) for d in data]
        
    if measure == "Phase_Ex":
        measure="EI"
    elif measure == "DFA_EI":
        measure = "DFA"
    elif measure == "AbsoluteAlpha":
        measure = "Alpha"
    data_dict[measure] = np.array(data)
    #data_dict[measure] = data
        

n_windows = 7
labels = get_region_labels()
Path("/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters").mkdir(parents=True, exist_ok=True)
for ii, measure in enumerate(measures):
    
    if measure == "Phase_Ex":
        measure="EI"
    elif measure == "DFA_EI":
        measure = "DFA"
    elif measure == "AbsoluteAlpha":
        measure = "Alpha"
    
    psd_avg_data = data_dict[measure].copy().mean(axis=0)
    
    for w_id in range(n_windows):
        Path(f"/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters/window_{w_id+1}").mkdir(parents=True, exist_ok=True)

        psd_profiles_dict = {}
        for key, value in zip(labels, psd_avg_data[:, w_id]):
            psd_profiles_dict[key] = value
        
        title=measure
        # This is just to change the title for the names of these two 
        # particular measures
        if measure == "Phase_Ex":
            title="EI"
        elif measure == "DFA_EI":
            title = "DFA"
        elif measure == "AbsoluteAlpha":
            title = "Alpha"
        fig = plot_parcellated_data(psd_profiles_dict)
        
        #if w_id == 0:
        #   fig.axes[0].set_title(title, size=20)
        #fig.axes[1].tick_params(labelsize=12)
        #fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
        #ticks = fig.axes[1].get_xticks()
        #fig.axes[1].set_xticks([ticks[0], ticks[-1]])
        #fig.axes[1].set_xticklabels(["min", "max"])
        fig.axes[1].remove()
        fig.savefig(
            f"/home/javi/Documentos/meg-excitability-landscape/supp/plots_clusters/window_{w_id+1}/brainplot_{measure}.png",
                    dpi=300, bbox_inches="tight")

n_windows = 7
n_subjects = 71
between_measures_corrs = []
for w_id in range(n_windows): 
    for ii, mx in enumerate(list(data_dict.keys())):
        for jj, my in enumerate(list(data_dict.keys())):
            if mx == my:
                continue
            if ii>jj:
                continue
            for sid in range(n_subjects):
    
                between_measures_corrs.append(
                    [w_id, 
                     f"{mx}-{my}", 
                     sid, 
                     spearmanr(data_dict[mx][sid,:, w_id], 
                               data_dict[my][sid,:, w_id])[0]
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