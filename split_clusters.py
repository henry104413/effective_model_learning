#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import sys # for passing command line arguments
#import pickle
import json
import numpy as np

# cluster assignment dictionary and best models must be in this folder
# for each cluster 


# import cluster assignment dictionary:
# experiment name and number of defects in model taken as command line arguments,
# or defaults set here if unavailable
try:
    experiment_name = str(sys.argv[1])
except:
    print('Models splitting:\n Using default experiment name in filename'
          +'\n - not command line argument')
    experiment_name = '250422_longer_run_Wit_Fig4b-grey'
try:
    defects_number = int(sys.argv[2])
except:
    print('Models splitting:\n Using default defects number in filename'
          +'\n - not command line argument')
    defects_number = 1
filename_base = experiment_name + '_D' + str(defects_number)
clustering_output_file = filename_base + '_clustering_output.json'
with open(clustering_output_file, 'r') as filestream:
    clusters = json.load(filestream)


# choose number of clusters:
# command line argument or default set here
try:
    clusters_count = int(sys.argv[3])
except:
    print('Models splitting:\n Using default number of clusters'
          +'\n - not command line argument')
    clusters_count = 2
if clusters_count not in [int(x) for x in clusters['assignments']]:
    raise RuntimeError('\nNo assignments available for this number of clusters!\n')
    
models = clusters['filenames']
models_count = len(models)
assignments = clusters['assignments'][str(clusters_count)]


# generate and save in appropriate file list of models for each cluster:
for i in range(clusters_count):
    
    ith_cluster = [models[j] for j in range(len(models)) if assignments[j] == i]
    
    # np.savetxt(filename_base + '_C' + str(i) + '.txt', 
    #            np.array(ith_cluster, dtype=str),
    #            fmt = '%.0d',
    #            delimiter = '\n',
    #            comments = '')

    with open(filename_base + '_C' + str(i) + '.txt', 'w') as filestream:
        for model in ith_cluster:
            filestream.write(model + '\n')
            
              
    

# clusters['clusters_counts'] is a bit redundant
# since it should be equal to clusters['assigments'].keys()

# also why is the clusters count as key in assignments being saved as a string??
# if changed, than change it here too!

# i think the assignment should just save the core filename
# without the _best.pickle