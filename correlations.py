#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)

Finds correlations for across selected clusters of models.
"""


import pandas as pd
import pickle
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import configs
import learning_model
from definitions import observable_shorthand2pretty as ops_longlabels, ops

# settings:
cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '251122-run' + '_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 2
Rs = [1,4,5,6,7,8,11,12,13,15,17,19,20]
clustering_name = 'clustering-sub100'
chosen_k = 7
Rs_tag = ''.join([str(x) + ',' for x in Rs])[:-1]
hyperparams = configs.get_hyperparams(config_name)
output_name = (experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_'
               + clustering_name + '_k' + str(chosen_k) + '_correlations_vol3')
correlation_hierarchical_clustering_thresholds = [0.7, 0.5]

# import lists of models in each cluster (currenlty not centres though),
# and example model (for parameter labels):
example_model_file = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(Rs[0]) + '_best.pickle'
with open(example_model_file, 'rb') as filestream:
    example_model = pickle.load(filestream)
models_by_clusters_file = (experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_'
            + clustering_name + '_k' + str(chosen_k) + '_combined_assignments.pickle')
with open(models_by_clusters_file, 'rb') as filestream:
    models_by_clusters = pickle.load(filestream)

_, labels, labels_latex = example_model.vectorise_under_library(hyperparameters = hyperparams)



#%%
# bar plot of clusters' popularity:
plt.figure()
plt.bar(list(models_by_clusters.keys()), [len(models_by_clusters[x]) for x in models_by_clusters],
        color = 'navy')
plt.xlabel('cluster')
plt.ylabel('number of models')



#%% also plot dynamics comparison for cluster centres and champions

if True: 
    
    # import dictionary of ts, sx, sy, sz observable values: 
    # (sx equal to original and rest simulated, all with noise with std = 0.01)    
    with open('simulated_250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle',
              'rb') as filestream:
        simulated_data = pickle.load(filestream)    
    ts, sx, sy, sz = [simulated_data[x] for x in ['ts', 'sx', 'sy', 'sz']]
            
    # target data:
    # (encapsulate datasets and corresponding observable labels into lists)
    measurement_datasets = [sx, sy, sz]
    measurement_observables = ['sigmax', 'sigmay', 'sigmaz']
    
    # ts for evaluating models at for comparison with target:
    evaluation_ts = ts

    
    # centres file (now pickle with centres for all ks - could also use specific k csv centres file)
    outputs_each_k_file = (experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_'
                + clustering_name + '_outputs_each_k.pickle')
    with open(outputs_each_k_file, 'rb') as filestream:
        outputs_each_k = pickle.load(filestream)
        clusters_centres = outputs_each_k[chosen_k]['centres']
    # note: clusters_centres is np array with rows for clusters and columns for parameters
    
    # turn into models and do dynamics comparison of centres vs target datasets: 
    centres_datasets = {'sx': [], 'sy': [], 'sz':[]}
    centre_model = learning_model.LearningModel()
    for c, centre in enumerate([list(clusters_centres[x,:]) for x in range(clusters_centres.shape[0])]):
        vectorised_centre = (centre, labels, labels_latex)
        centre_model.configure_to_params_vector(vectorised_centre, 
                                                D = D,  
                                                qubit_initial_state = ops['plus'],
                                                defect_initial_state = ops['mm'])
        temp = centre_model.calculate_dynamics(evaluation_ts, observable_ops = measurement_observables)
        for o, op in enumerate(centres_datasets.keys()):
            centres_datasets[op].append(temp[o])
            plt.figure()
            plt.xlabel('t (us)')
            plt.ylabel(ops_longlabels[op])
            plt.ylim([-1, 1])
            #plt.plot(ts, simulated_data[op], 'b.', markersize = 1, label = 'target')
            plt.errorbar(ts, simulated_data[op], yerr = 0.01, fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target')
            plt.plot(evaluation_ts, centres_datasets[op][c], 
                     'r-', linewidth = 1, alpha = 0.7, label = 'model')
            plt.legend()
            plt.title('cluster centre ' + str(c))
            plt.savefig(output_name + '_C' + str(c) + '_centre_' + str(op) + '_comparison' + '.svg', 
                        dpi = 1000, bbox_inches='tight')
        
    # also do the same for cluster champions - find, turn into models and do comparison vs target datasets: 
    # note: currently based on posterior (can also do loss):
    source_base = experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_' + clustering_name
    with open(source_base + '_points.pickle', 'rb') as filestream:
        points = pickle.load(filestream)
    with open(source_base + '_posteriors.pickle', 'rb') as filestream:
        posteriors = pickle.load(filestream)
    assignments = list(outputs_each_k[chosen_k]['assignments'])
    
    champions = {}
    champ_posteriors = {}
    champion_model = learning_model.LearningModel()
    champions_datasets = {'sx': [], 'sy': [], 'sz':[]}
    for c in range(chosen_k):
    
        # find highest posterior and corresponding parameter vector (point) for each cluster assignment c
        champion, posterior, assignment = max(filter(lambda x: x[2] == c, zip(points, posteriors, assignments)),
                                            key = lambda x: x[1])
        champions[assignment] = champion
        champ_posteriors[assignment] = posterior
        
        vectorised_champion = (champion, labels, labels_latex)
        champion_model.configure_to_params_vector(vectorised_champion, 
                                                D = D,  
                                                qubit_initial_state = ops['plus'],
                                                defect_initial_state = ops['mm'])
        temp = champion_model.calculate_dynamics(evaluation_ts, observable_ops = measurement_observables)
        for o, op in enumerate(champions_datasets.keys()):
            champions_datasets[op].append(temp[o])
            plt.figure()
            plt.xlabel('t (us)')
            plt.ylabel(ops_longlabels[op])
            plt.ylim([-1, 1])
            plt.errorbar(ts, simulated_data[op], yerr = 0.01, fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target')
            plt.plot(evaluation_ts, champions_datasets[op][c], 
                     'r-', linewidth = 1, alpha = 0.7, label = 'model')
            plt.legend()
            plt.title('cluster champion ' + str(c))
            plt.savefig(output_name + '_C' + str(c) + '_champion_' + str(op) + '_comparison' + '.svg',
                        dpi = 1000, bbox_inches='tight')
        
    # overall champion:
    # champion, posterior = max(zip(points, posteriors), key = lambda x: x[1])    
    
        
#'251122-run_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_Rs1,4,5,6,7,8,11,12,13,15,17,19,20_clustering-sub100_clustering_centres.pickle'

#%%
# find correlations, dendrograms, and clustered parameters in selected cluster combinations

# list of lists, each inner list for combinations of cluster to analyse together: 
cluster_combinations = []
# each separately:
cluster_combinations.extend([x] for x in (models_by_clusters.keys()))
# all together: 
cluster_combinations.append(list(list(models_by_clusters.keys())))
 
for cluster_combination in cluster_combinations:
    # find and save correlations for this cluster combination
    Cs_tag = ''.join([str(x) + ',' for x in cluster_combination])[:-1]
    
    # collect all present values for each parameter: 
    parameter_lists = {name: [] for name in labels} 
    # note: key for each parameter, each entry then list of values of that parameter across relevant models
    for cluster in cluster_combination:
    # go over all clusters in this cluster combination
        this_cluster_models = models_by_clusters[cluster]
        for model in this_cluster_models: 
        # go over all models in this cluster
            for j, value in enumerate(model):
                # go over all paramters of this model and add each to list of all values of that parameter
                parameter_lists[labels[j]].append(model[j])
    
    # parameter values dataframe, as well as correlation, dissimilarity, and linkage matrices:            
    data = pd.DataFrame(data = parameter_lists, columns = labels[:]) # columns! ?? is it still now without energies??
    CM = data.corr()
    dissimilarity = 1 - abs(CM)
    Z = linkage(squareform(dissimilarity), 'complete')
    
    # plot correlation matrix:
    plt.figure(figsize=(10,10))
    seaborn.heatmap(CM, annot=False, cmap=cmap, fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
    # colormaps: coolwarm, PiYG
    plt.savefig(output_name + '_Cs' + Cs_tag + '_matrix.svg', dpi = 1000, bbox_inches='tight')
    
    # hierarchical clustering:
    
    # plot dendrogram:    
    plt.figure(figsize=(10,5))
    dendrogram(Z, labels=data.columns, orientation='top', 
               leaf_rotation=90);
    plt.ylabel('correlation')
    plt.savefig(output_name + '_Cs' + Cs_tag + '_dendrogram.svg', dpi = 1000, bbox_inches='tight')
    
    # plot correlation matrix with clustered parameters
    # note: now clustering parameters not models!!
    for threshold in correlation_hierarchical_clustering_thresholds:
        flattened_dendrogram_labels = fcluster(Z, threshold, criterion='distance')
        
        # indices to sort labels
        labels_order = np.argsort(flattened_dendrogram_labels)
        
        # new dataframe with sorted columns
        for idx, i in enumerate(data.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(data[i])
            else:
                df_to_append = pd.DataFrame(data[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
        
        # plot:        
        plt.figure(figsize=(10,10))
        clust_CM = clustered.corr()
        # seaborn.heatmap(round(clust_CM,2), cmap='RdBu', annot=True, annot_kws={"size": 7}, vmin=-1, vmax=1);
        seaborn.heatmap(clust_CM, cmap=cmap, annot=False, annot_kws={"size": 7}, vmin=-1, vmax=1);
        plt.savefig(output_name + '_Cs' + Cs_tag + '_thr' + str(threshold) + '_clustered_matrix.svg', dpi = 1000, bbox_inches='tight')
    
    
