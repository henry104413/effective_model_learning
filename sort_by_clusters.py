#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import sys # for passing command line arguments
import json
import pickle

# note: cluster assignment dictionary and best models must be in this folder
# command line arguments: experiment name, defects number, number of clusters (k)


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


# generate and save in appropriate file a list of models for each cluster:
# (as bare model names ending with their repetition index)
# note: !! extra first line with cluster champion (later repeated in its original place)
for i in range(clusters_count):
    
    # list of all models in cluster
    ith_cluster = [models[j] for j in range(len(models)) if assignments[j] == i]
    
    # extract losses for each model from pickle files:
    cluster_losses = []
    for model in ith_cluster:
        with open(model + '_best.pickle', 'rb') as filestream:
            full_model = pickle.load(filestream)
            cluster_losses.append(full_model.final_loss)
    
    # find model with lowest loss:
    cluster_champion = ith_cluster[cluster_losses.index(min(cluster_losses))]
    
    # output to file:
    with open(filename_base + '_Cs' + str(clusters_count)
              + '_C' + str(i) + '.txt', 'w') as filestream:
        filestream.write(cluster_champion + '\n')
        for model in ith_cluster:
            filestream.write(model + '\n')
            

# to do: start each list with name of BEST model in current cluster?
            
              
 # to do: then a bash (or python?) script to use these lists to put corresponding files together
   

# clusters['clusters_counts'] is a bit redundant
# since it should be equal to clusters['assigments'].keys()

# also why is the clusters count as key in assignments being saved as a string??
# if changed, than change it here too!
# it is being saved as int so maybe a json thing that it changes them to string upon dumping

# i think the assignment should just save the core filename
# without the _best.pickle
