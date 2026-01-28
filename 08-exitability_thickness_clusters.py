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
sys.path.append("/home/javi/Documentos/meg-excitability-clustering/src")
from plots import  plot_with_overlays
from input_data import get_region_labels, load_data
from netneurotools import stats

# ================================
# 2. Load and Prepare data
# ================================

labels = get_region_labels()
psd_data, strategies = load_data(ranked=True)
headers = psd_data.keys()
psd_data = np.array(list(psd_data.values()))

measure_dat = {}
for ii, meas in enumerate(strategies):
    measure_dat[meas] = pd.DataFrame(psd_data[:, ii, :], columns=labels, index=headers)
    

subjects = pd.DataFrame({"Subject":[header.split("_")[0] for header in headers]})
# BUT!
# ASP1S001 and NP806 and Subject06 are the same subject with same MRI
idxs = subjects[(subjects.Subject == "ASP1S001") |  (subjects.Subject == "NP806")].index 
subjects.loc[idxs, "Subject"] = "Subject06"
# NO802 same subject as Subject01 with same MRI
idxs = subjects[(subjects.Subject == "NP802")].index 
subjects.loc[idxs, "Subject"] = "Subject01"
# NP808 and Subject09 are the same with same MRI
idxs = subjects[(subjects.Subject == "NP808")].index 
subjects.loc[idxs, "Subject"] = "Subject09"

# Load thickness data. Convert observed spatial values into ranks as we 
# are going to be doing spearman correlations
thick_dat = {}
for header in headers:
    file = glob(f"/home/javi/Documentos/meg-excitability-clustering/data/structure/Thickness/{header}*")[0]
    thick_dat[header]  = rankdata(loadmat(file, struct_as_record=False, 
                                          squeeze_me=True)["Output"].AtlasData[2:,0])
thick_dat_df = pd.DataFrame(thick_dat).T    
thick_dat_df.columns = labels

# Load Excitability data, in the same order as the thickness data
strategies = [os.path.basename(folder) for folder 
            in glob("/home/javi/Documentos/meg-excitability-clustering/data/AllExcitability/*")]

    
# Load cluster info    
clus_res = np.load(
    "/home/javi/Documentos/meg-excitability-clustering/data/clusters_new.npz")

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


# ================================
# 3. Correlate cluster-wise data with thickness
# ================================


corrs_ex_thick = {}
for strategy in sorted(clus_measure_dat.keys()):
    corrs = []
    for header in headers:
        corrs.append(pearsonr(clus_measure_dat[strategy].loc[header,:], 
                              thick_dat_df.loc[header,:])[0])
    corrs_ex_thick[strategy] = corrs
    
corrs_ex_thick = pd.DataFrame(corrs_ex_thick)
corrs_ex_thick["Subject"] = subjects["Subject"]


# Group-level significance
spins = np.load("/home/javi/Documentos/meg-excitability-clustering/data/spin_permutations.npy")
nspins = spins.shape[-1]

n_clus = len(np.unique(clus_res["clus_id"]))
corrs_dict = dict()
for measure in clus_measure_dat.keys():
    
    ei_clus = clus_measure_dat[measure].mean(0).to_numpy()
    thick_data = thick_dat_df.mean(0).to_numpy()
    
    corrs_dict[measure] = stats.permtest_pearsonr(ei_clus, thick_data,
                                        #ei_data_ordered[:,jj], 
                                        n_perm=nspins,
                                        resamples=spins, seed=1234)



n_clus = len(np.unique(clus_res["clus_id"]))
corrs_obs_dict = dict()
for measure in clus_measure_dat.keys():

    cors_obs = []
    for index in clus_measure_dat[measure].index:
        ei_clus = clus_measure_dat[measure].loc[index,:]
        thick_data = thick_dat_df.loc[index,:]
        cors_obs.append(stats.efficient_pearsonr(ei_clus, thick_data)[0])
    corrs_obs_dict[measure] = np.mean(cors_obs)

corrs_perm_dict = dict()
# Create empty vector with permutation-based correlations
for measure in clus_measure_dat.keys():
    corrs_perm_dict[measure] = np.zeros(nspins)

# iterate for each permuation
for perm in tqdm(range(nspins)):
    perm_ix = spins[:, perm]
    # Iterate through measures
    for measure in clus_measure_dat.keys():
        cor_perm = []
        # Iterate through subjects, storing the permutation-based correlation 
        for index in clus_measure_dat[measure].index:
            ei_clus = clus_measure_dat[measure].loc[index,:].to_numpy()
            thick_data = thick_dat_df.loc[index,:].to_numpy()[perm_ix]
            cor_perm.append(stats.efficient_pearsonr(ei_clus, thick_data)[0])
        # The group-based correlation is the average
        corrs_perm_dict[measure][perm] = np.mean(cor_perm)

        
    
pvals_subject_dict = dict()
for measure in clus_measure_dat.keys():
    pvals_subject_dict[measure] = (1 + sum(abs(corrs_perm_dict[measure]) > abs(corrs_obs_dict[measure])))/(1+nspins)
        
        

# ===================================================
# 4. region-wise association thickness and clusters #
# ===================================================

n_regs=100
zstats_dict, pvals_dict = dict(), dict()
subjects_df = pd.DataFrame({"Subject": thick_dat_df.index})

#X_reg = pd.DataFrame(X_clus[:,:,ireg], columns = subset_strategies)
for xx, col_x in enumerate(sorted(clus_measure_dat.keys())):
    zstats = dict() # np.zeros(n_regs)
    pvals = dict()  # np.ones(n_regs)
    
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
       if model_fit.converged is False:
           continue
       # Calculating explicitly the z-stat
       # This should be equal to the tvalue attribute, but
       # jst in case
       coef = model_fit.fe_params["thickness"]
       se = np.sqrt(model_fit.cov_params().loc["thickness", "thickness"])            
       z = coef/se
    
       zstats[reg] = z
       pvals[reg] = model_fit.pvalues["thickness"]

    zstats_dict[col_x] = zstats
    pvals_dict[col_x] = pvals




pvals_melt_df = pd.melt(pd.DataFrame(pvals_dict), value_name="pval", 
                        var_name="strategy", ignore_index=False)
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

out_dir = "/home/javi/Documentos/meg-excitability-clustering/plots"
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

