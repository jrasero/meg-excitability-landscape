
"""
Script that analyzes the association between thickness 
and Excitability clusters

"""

# ================================
# 1. Imports
# ================================

from scipy.io import loadmat
from glob import glob
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pylab as plt
import os
from scipy.stats import rankdata
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from os.path import join as opj
import sys
import warnings
sys.path.append("/home/javi/Documentos/meg-excitability-landscape/src")
from plots import  plot_with_overlays
from input_data import get_region_labels, load_data
from netneurotools import stats

# ================================
# 2. Load and Prepare data
# ================================

labels = get_region_labels()
data, measures, subjects = load_data(ranked=True)
headers = data.keys()
data = np.array(list(data.values()))

measure_dat = {}
for ii, meas in enumerate(measures):
    measure_dat[meas] = pd.DataFrame(data[:, ii, :], columns=labels, index=headers)
    
thick_dat = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_main.npz")["thickness"]

thick_dat = np.column_stack([rankdata(thick_dat[six, :]) for six in range(thick_dat.shape[0])])
thick_dat_df = pd.DataFrame(thick_dat.T)    
thick_dat_df.columns = labels
thick_dat_df.index = headers


# Load cluster info    
clus_res = np.load(
    "/home/javi/Documentos/meg-excitability-landscape/data/clusters_new.npz")

for iclus in range(len(np.unique(clus_res["clus_id"]))):
    print(f"clus {iclus} is formed by", clus_res["labels"][clus_res["clus_id"] == iclus])
       
# Average the within clusters
clus_measure_dat = {}

clus_measure_dat["40Hz"] = 0.5*(measure_dat["40Hz"] + measure_dat["40Hz_ZScore"])
clus_measure_dat["Alpha"] = 1/3*(measure_dat["AbsoluteAlpha"] + 
                                 measure_dat["AlphaRelative"] + 
                                 measure_dat["FOOOF_Alpha"])
clus_measure_dat["DFA"] = measure_dat["DFA_EI"]
clus_measure_dat["Exponent"] =  0.5*(measure_dat["Exponent"] + measure_dat["Offset"])
clus_measure_dat["Phase_Ex"] =  measure_dat["Phase_Ex"]
clus_measure_dat["SE"] =  measure_dat["SE"]


n_regs=100
zstats_dict, pvals_dict = dict(), dict()
subjects_df = pd.DataFrame({"Subject": subjects})

for xx, col_x in enumerate(sorted(clus_measure_dat.keys())):
    zstats = dict()
    pvals = dict()  
    
    for ireg, reg in tqdm(enumerate(thick_dat_df.columns)):
        
       
       X_reg = pd.DataFrame(np.column_stack((
           [clus_measure_dat[col_x].loc[:, reg].to_numpy(),
            thick_dat_df.loc[:, reg].to_numpy()])), 
                            columns = [col_x, "thickness"])
       
       data_df = pd.concat([X_reg, subjects_df], axis=1)
       
       model = smf.mixedlm(f"Q('{col_x}')~thickness", data=data_df, 
                        groups="Subject")
       
    
       with warnings.catch_warnings():
           warnings.simplefilter("ignore")
           model_fit = model.fit(method='powell', maxiter=1000, reml=False)
       
           # Calculating explicitly the z-stat
           # This should be equal to the tvalue attribute, but
           # jst in case
           coef = model_fit.fe_params["thickness"]
           se = np.sqrt(model_fit.cov_params().loc["thickness", "thickness"])            
           z = coef/se
           
           pv = model_fit.pvalues["thickness"]
           
           if model_fit.converged is False:
               z=0
               pv=1
       zstats[reg] = z
       pvals[reg] = pv

    zstats_dict[col_x] = zstats
    pvals_dict[col_x] = pvals


pvals_melt_df = pd.melt(pd.DataFrame(pvals_dict), value_name="pval", 
                        var_name="strategy", ignore_index=False)
pvals_melt_df.pval = pvals_melt_df.pval.fillna(1)
pvals_melt_df["pval_fdr"] = multipletests(pvals_melt_df.pval, 
                                          method="fdr_bh")[1]

for col_x in sorted(clus_measure_dat.keys()):
    print(col_x, 
          sum(pvals_melt_df[pvals_melt_df.strategy==col_x].pval_fdr < 0.05))


print(pvals_melt_df[pvals_melt_df.pval_fdr < 0.05])

labels = get_region_labels()


# ===================================================
# 5. Brainplots for these region-wise associations  #
# ===================================================

out_dir = "/home/javi/Documentos/meg-excitability-landscape/plots"
Path(out_dir).mkdir(exist_ok=True, parents=True)

for case in zstats_dict.keys():
    
    zstat_reg_sig_dict = {}
    zstat_reg_all_dict = {}
    outline_dict = {}
    
    pvals_case = pvals_melt_df[pvals_melt_df.strategy==case]
    zstat_case = pd.DataFrame(zstats_dict).loc[:,[case]]
    zstat_case.columns = ["zstat"]
    data_plot = pd.concat([zstat_case, pvals_case], axis=1, join="inner")
    for reg, row in data_plot.iterrows():
        
    #for key, z, pv in zip(labels, zstats, pvals):
        zstat_reg_all_dict[reg] = row.zstat
        outline_dict[reg] = int(row.pval_fdr<0.05) 
        zstat_reg_sig_dict[reg] = row.zstat*(row.pval_fdr<0.05)
        
    #vmax = max(abs(data_plot.zstat))
    fig = plot_with_overlays(zstat_reg_all_dict, outline_dict = outline_dict, 
                             cmap="RdBu",
                             conn_overlay_dict=zstat_reg_sig_dict, 
                             alpha=0.5, vmin=-5, vmax=5)
   
    fig.axes[0].set_title(case, size=30)
    fig.axes[1].tick_params(labelsize=20)
    fig.axes[1].set_position([0.25, -1.8, 0.5, 2])
    ticks = fig.axes[1].get_xticks()
    fig.axes[1].set_xticks([ticks[0], ticks[-1]])
        
    fig.savefig(opj(out_dir, f"thickness_{case}.png"), dpi=300)
    fig.savefig(opj(out_dir, f"thickness_{case}.svg"), dpi=300)
    plt.close(fig)

