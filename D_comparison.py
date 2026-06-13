#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:21:42 2026

@author: henry
"""




import pickle
import numpy as np
    
#%%

# 1) have dictionary with keys for dataset then for D containing
# - to contain maybe the moments of the accepted pseudoposteriors
# maybe try with full list and if it's too large then just combine the moments...
# 2) another dictionary with feature size for each key
# 

# settings and source data:
experiment_name_base = '260516-exp'
experiment_name = experiment_name_base
noise_stdev = float(0.0) # assumed same = 0 for all these
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'

og_sources = ['Wit-Fig4-5-0_1',
              'Wit-Fig4-6-0_025',
              'Wit-Fig4-6-0_1',
              'Wit-Fig4-6-0_2',
              'Wit-Fig4-7-0_1']
Ds = [1,2,3] # assumed same for each og_file
Rs = [i+1 for i in range(3)] # assumed same for each combination of D and og_file

experiment_name += '_std' + str(noise_stdev).replace('.','p')

# nested dictionaries with first key for dataset, second key for D
pps = {} # pseudoposteriors - list to iteratively add data from all chains
pps_np = {} # pseudoposteriors - converted numpy arrays for later analysis (populating np arrays dynamically is likely ineffiecient)
pps_np_proportion = {}
pps_np_proportion_mean = {}
pps_np_proportion_std = {}

for og_source in og_sources:

    # prepare subdictionary for populating
    pps[og_source] = {}
    pps_np[og_source] = {}
    pps_np_proportion[og_source] = {}
    pps_np_proportion_mean[og_source] = {}
    pps_np_proportion_std[og_source] = {}

    for D in Ds:
        
        # combine pseudoposteriors from all chains (Rs)
        pps[og_source][D] = []
        for R in Rs:
            # import pickle, which is a list of floats
            filename = experiment_name + '_' + og_source + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_accepted_log_posterior.pickle'
            with open(filename, 'rb') as filestream:
                temp = pickle.load(filestream)
                pps[og_source][D].extend(temp)
                
        # turn to numpy arrays, specify and select proportion, and compute mean and stdev for each datafile and D:
        pps[og_source][D].sort()
        pps_np[og_source][D] = np.array(pps[og_source][D])                
        # take BIG EFFING CARE HERE!! .sort() does NOT return so pps[og_source][D].sort() is NOT the sorted list, it is None!!!
        # note: now sorted in ascending order - starting at most NEGATIVE values!!!
        proportion = 1 # ratio to take from start of array (smallest/most negative elements)
        pps_np_proportion[og_source][D] = pps_np[og_source][D][0:round(proportion*len(pps_np[og_source][D]))]
        pps_np_proportion_mean[og_source][D] = pps_np_proportion[og_source][D].mean()
        pps_np_proportion_std[og_source][D] = pps_np_proportion[og_source][D].std()
        
        
        
        
        
        
                



#%%

