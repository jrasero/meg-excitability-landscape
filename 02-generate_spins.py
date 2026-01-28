"""
Script to generate the spin permutations to be later used

"""

import numpy as np
import pandas as pd
import sys
sys.path.append("/home/javi/Documentos/meg-excitability-clustering/src")
from input_data import get_region_labels
from nilearn.datasets import fetch_atlas_schaefer_2018
from netneurotools import stats

# Get region labels
labels = get_region_labels()


# ----- Coordinates, for spatial null maps -----
schaefer = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
coords = pd.read_csv(
    "https://raw.githubusercontent.com/ThomasYeoLab/"+\
        "CBIG/refs/heads/master/stable_projects/brain_parcellation/"+\
            "Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/"+\
                "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
            )
coords.index = coords.iloc[:,1]
# Order according to hansen's repo, just in case
coords = coords.loc[schaefer["labels"].astype(str),:]

# ----- Spatially rearrange data -----
# All these data are not spatially arranged as our measurements maps,
# so we have to make sure thet match region by region

# Step 1: Create a mapping from string to its order in `b`
order = {val: i for i, val in enumerate(labels)}

# Step 2: Get the indices that would sort `a` according to `b`
indices = np.argsort([order[val] for val in schaefer["labels"].astype(str)])

# Step 3: Apply these indices to both coordinates and receptors data
coords_ordered = coords.iloc[indices,:]

# Create spins samples
nnodes = 100
hemiid = (coords_ordered.R > 0).astype(int).to_numpy()
nspins = 10000
spins = stats.gen_spinsamples(
    coords_ordered.loc[:, ["R", "A","S"]].to_numpy(), 
    hemiid, n_rotate=nspins, method="vasa", 
    seed=1234)
np.save("/home/javi/Documentos/meg-excitability-clustering/data/spin_permutations.npy", 
        spins)