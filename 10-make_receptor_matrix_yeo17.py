"""
Concatenate parcellated PET images into region x receptor matrix of densities.
Again, we need to do this because our schaefer corresponds to the 17 Yeo 
Networks, in contrast to Hansen, et al.  Nat Neurosci 25, 1569–1581 (2022).
Also note, that one of the maps for VAChT was not used because it is not 
available in the original Hansen2022 publication. 

"""

import numpy as np
from netneurotools import datasets, plotting
from matplotlib.colors import ListedColormap
from scipy.stats import zscore
from nilearn.datasets import fetch_atlas_schaefer_2018

scale = 'scale100'

schaefer = fetch_atlas_schaefer_2018(n_rois=100)
nnodes = len(schaefer['labels'])
path = "/home/javi/Documentos/meg-excitability-clustering/data/receptors/"+scale+"/"

# concatenate the receptors
receptors_csv = ['/5HT1a_way_hc36_savli.csv',
                 '/5HT1b_p943_hc22_savli.csv',
                 '/5HT1b_p943_hc65_gallezot.csv',
                 '/5HT2a_cimbi_hc29_beliveau.csv',
                 '/5HT4_sb20_hc59_beliveau.csv',
                 '/5HT6_gsk_hc30_radhakrishnan.csv',
                 '/5HTT_dasb_hc100_beliveau.csv',
                 '/A4B2_flubatine_hc30_hillmer.csv',
                 '/CB1_omar_hc77_normandin.csv',
                 '/D1_SCH23390_hc13_kaller.csv',
                 '/D2_flb457_hc37_smith.csv',
                 '/D2_flb457_hc55_sandiego.csv',
                 '/DAT_fpcit_hc174_dukart_spect.csv',
                 '/GABAa-bz_flumazenil_hc16_norgaard.csv',
                 '/H3_cban_hc8_gallezot.csv', 
                 '/M1_lsn_hc24_naganawa.csv',
                 '/mGluR5_abp_hc22_rosaneto.csv',
                 '/mGluR5_abp_hc28_dubois.csv',
                 '/mGluR5_abp_hc73_smart.csv',
                 '/MU_carfentanil_hc204_kantonen.csv',
                 '/NAT_MRB_hc77_ding.csv',
                 '/NMDA_ge179_hc29_galovic.csv',
                 #'/VAChT_feobv_hc3_spreng.csv', This was not available in Hansen2022
                 '/VAChT_feobv_hc4_tuominen.csv',
                 '/VAChT_feobv_hc5_bedard_sum.csv',
                 '/VAChT_feobv_hc18_aghourian_sum.csv']

# combine all the receptors (including repeats)
r = np.zeros([nnodes, len(receptors_csv)])
for i in range(len(receptors_csv)):
    r[:, i] = np.genfromtxt(path + receptors_csv[i], delimiter=',')

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
np.save(path+'receptor_names_pet.npy', receptor_names)

# make final region x receptor matrix
receptor_data = np.zeros([nnodes, len(receptor_names)])
receptor_data[:, 0] = r[:, 0] # 5HT1a

# weighted average of 5HT1B p943
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

#'/5HT2a_cimbi_hc29_beliveau.csv', '/5HT4_sb20_hc59_beliveau.csv', 
#'/5HT6_gsk_hc30_radhakrishnan.csv',
# '/5HTT_dasb_hc100_beliveau.csv',
# '/A4B2_flubatine_hc30_hillmer.csv',
# '/CB1_omar_hc77_normandin.csv',
# '/D1_SCH23390_hc13_kaller.csv']
receptor_data[:, 2:9] = r[:, 3:10]

# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

#array(['D2', 'DAT', 'GABAa', 'H3', 'M1'], dtype='<U6')
#['/DAT_fpcit_hc174_dukart_spect.csv',
# '/GABAa-bz_flumazenil_hc16_norgaard.csv',
# '/H3_cban_hc8_gallezot.csv',
# '/M1_lsn_hc24_naganawa.csv']
receptor_data[:, 10:14] = r[:, 12:16]

# weighted average of mGluR5 ABP688
#['/mGluR5_abp_hc22_rosaneto.csv',
# '/mGluR5_abp_hc28_dubois.csv',
# '/mGluR5_abp_hc73_smart.csv']
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

#array(['MOR', 'NET', 'NMDA'], dtype='<U6')
#['/MU_carfentanil_hc204_kantonen.csv',
# '/NAT_MRB_hc77_ding.csv',
# '/NMDA_ge179_hc29_galovic.csv']
receptor_data[:, 15:18] = r[:, 19:22]

# weighted average of VAChT FEOBV
#['/VAChT_feobv_hc4_tuominen.csv',
# '/VAChT_feobv_hc5_bedard_sum.csv',
# '/VAChT_feobv_hc18_aghourian_sum.csv']
receptor_data[:, 18] = (#zscore(r[:, 22])*3 
                        zscore(r[:, 22])*4 + zscore(r[:, 23])*5 + zscore(r[:, 24]))*18 / \
                       (4+5+18)#(3+4+5+18)

np.savetxt(path+'receptor_data_'+scale+'.csv', receptor_data, delimiter=',')


# ================================
# Plot Receptor data
# ================================

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

# plot each receptor map
if scale == 'scale100':
    annot = datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
    for k in range(len(receptor_names)):
        brain = plotting.plot_fsaverage(data=receptor_data[:, k],
                                        lhannot=annot.lh,
                                        rhannot=annot.rh,
                                        colormap='plasma',
                                        views=['lat', 'med'],
                                        data_kws={'representation': "wireframe"})
        brain.save_image(path+'figures/schaefer100/surface_receptor_'+receptor_names[k]+'.png')