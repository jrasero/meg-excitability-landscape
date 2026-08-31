"""
Auxiliary functions to import data
"""

from scipy.io import loadmat

def get_region_labels(n_rois=100):

    labels = loadmat(
        "/home/javi/Documentos/meg-excitability-landscape/data/Schaefer_labels_17net.mat")[f"roi{n_rois}"][0]
    
    #labels = [scout.Label for scout in scouts[2:]] # Take out medial wall
    labels = [l[0] for l in labels]
    
    for ii, label in enumerate(labels):
        if label.endswith("L"):
            labels[ii] = "17Networks_LH_" + label.split(" L")[0]
        else:
            labels[ii] = "17Networks_RH_" + label.split(" R")[0]
            
    return labels

def load_data(ranked=False):
    import numpy as np
    from scipy.stats import rankdata

    data_all = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_main.npz")["data"]
    measures = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_main.npz")["measures"]
    subjects = np.load("/home/javi/Documentos/meg-excitability-landscape/data/data_main.npz")["subjects"]
    
    
    data_mats = dict()
    for six, subj in enumerate(subjects):
        data_profiles = []
        for mix, meas in enumerate(measures):
            
            data = data_all[six, mix, :]
            if ranked:
                data = rankdata(data)
                
            data_profiles.append(data)
            
        data_mats[f"scan{six}"] = np.row_stack(data_profiles)
        
    return data_mats, measures, subjects
