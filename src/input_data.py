"""
Auxiliary functions to import data
"""

from scipy.io import loadmat

def get_region_labels():

    scouts = loadmat("/home/javi/Documentos/meg-excitability-clustering/data/scout_Schaefer_100_17net_102.mat", 
                     struct_as_record=False,  squeeze_me=True)["Scouts"]
    
    labels = [scout.Label for scout in scouts[2:]] # Take out medial wall
    
    for ii, label in enumerate(labels):
        if label.endswith("L"):
            labels[ii] = "17Networks_LH_" + label.split(" L")[0]
        else:
            labels[ii] = "17Networks_RH_" + label.split(" R")[0]
            
    return labels

def load_data(ranked=False):
    import numpy as np
    import os
    from glob import glob
    from scipy.stats import rankdata


    strategies = [os.path.basename(folder) for folder 
                in glob("/home/javi/Documentos/meg-excitability-clustering/data/AllExcitability/*")]
    
    check_strategy_equal = []
    for strategy in strategies:
        check_strategy_equal.append(len(glob(f"/home/javi/Documentos/meg-excitability-clustering/data/AllExcitability/{strategy}/*")))
    
    assert np.all(np.array(check_strategy_equal)==72)
    
    subjects = [os.path.basename(file).split("_")[0] 
                for file in sorted(glob("/home/javi/Documentos/meg-excitability-clustering/data/AllExcitability/AlphaRelative/*.mat"))]
    
    subjects[42:] = [name + "_REAL" if ii % 2 == 0 else name + "_SHAM" for ii, 
                     name in enumerate(subjects[42:])]
    
    check_all = []
    for subj in subjects:
       check_all.append(
           len(glob(f"/home/javi/Documentos/meg-excitability-clustering/data/AllExcitability/*/*{subj}*")))
       
    assert np.all(np.array(check_all)==10)
    
    psd_mats = dict()
    for subj in subjects:
        psd_profiles = []
        for strategy in sorted(strategies):
            mat_file = glob(
                f"/home/javi/Documentos/meg-excitability-clustering/data/AllExcitability/{strategy}/*{subj}*")[0]
            data = loadmat(mat_file, squeeze_me=True)["Output"]["AtlasData"].item()
            if data.ndim==2:
                data = data[:,0]
                
            data = data[2:] # Remove medial values
            ## NEW to align what excitability means...
            if strategy in ["FOOOF_Alpha", "AbsoluteAlpha", "AlphaRelative", 
                            "SE", "Exponent", "Offset"]:
                data = data.max() -  data + data.min()
                
            if ranked:
                data = rankdata(data)
                
            psd_profiles.append(data)
            
        psd_mats[subj] = np.row_stack(psd_profiles)
        
    return psd_mats, sorted(strategies)
