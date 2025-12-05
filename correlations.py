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
import random
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import configs
import learning_model
from definitions import observable_shorthand2pretty as ops_longlabels, ops

# settings:
cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '251204-LN' + '_Wit-Fig4-6-0_025' # including experiment base and source file name
simulated_std = 0.1
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 2
Rs = [1,2,3,4,5] # for D2
#Rs = [x+1 for x in range(30)]
Rs_tag = ''.join([str(x) + ',' for x in Rs])[:-1]
clustering_name = 'e100'
chosen_k = 4
hyperparams = configs.get_hyperparams(config_name)
output_name = (experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_'
               + clustering_name + '_k' + str(chosen_k) + '_corr')
correlation_hierarchical_clustering_thresholds = [0.7, 0.5]
target_data_pickle_file = (
    'simulated-std' + str(simulated_std).replace('.', 'p')
    + '_250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle.py')
# note: using naming convention for std

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

# also import points, assignments, and outputs_each_k for centres below:
source_base = experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_' + clustering_name
outputs_each_k_file = (experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_'
            + clustering_name + '_outputs_each_k.pickle')
with open(source_base + '_points.pickle', 'rb') as filestream:
    points = pickle.load(filestream)
with open(source_base + '_posteriors.pickle', 'rb') as filestream:
    posteriors = pickle.load(filestream)
with open(outputs_each_k_file, 'rb') as filestream:
    outputs_each_k = pickle.load(filestream)
assignments = list(outputs_each_k[chosen_k]['assignments'])



#%%
# bar plot of clusters' popularity:
plt.figure()
plt.bar(list(models_by_clusters.keys()), [len(models_by_clusters[x]) for x in models_by_clusters],
        color = 'navy')
plt.xlabel('cluster')
plt.xticks(list(range(chosen_k)), labels = [str(x) for x in list(range(chosen_k))])
plt.ylabel('number of models')
plt.savefig(output_name + '_cluster_popularity' + '.svg', 
            dpi = 1000, bbox_inches='tight')



#%% also plot dynamics comparison for cluster centres and champions

if not True: 
    
    # import dictionary of ts, sx, sy, sz observable values: 
    # (sx equal to original and rest simulated, all with noise with std = 0.01)    
    with open('simulated_250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle.py',
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
            plt.errorbar(ts, simulated_data[op], yerr = simulated_std, fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target')
            plt.plot(evaluation_ts, centres_datasets[op][c], 
                     'r-', linewidth = 1, alpha = 0.7, label = 'model')
            plt.legend()
            plt.title('cluster centre ' + str(c))
            plt.savefig(output_name + '_C' + str(c) + '_centre_' + str(op) + '_comparison' + '.svg', 
                        dpi = 1000, bbox_inches='tight')
        
    # also do the same for cluster champions - find, turn into models and do comparison vs target datasets: 
    # note: currently based on posterior (can also do loss):
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
            plt.errorbar(ts, simulated_data[op], yerr = simulated_std, fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target')
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
cluster_combinations = [[3]]
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
    plt.savefig(output_name + '_Cs' + Cs_tag + '_mat.svg', dpi = 1000, bbox_inches='tight')
    
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
        plt.savefig(output_name + '_Cs' + Cs_tag + '_thr' + str(threshold) + '_cmat.svg', dpi = 1000, bbox_inches='tight')
    
    
    
# %%
# popularity of different processes:
# manually chosen sets for now... cheeky bit of code

cluster_choices = [1,2,3,0]
# formatting - must have options for at least each cluster choice, can be longer:
colours = ['red', 'blue', 'black', 'purple']
heights = [0.8, 0.5, 0.3, 0.1] # ideally descending
alphas = [0.3, 0.3, 0.3, 1] # ideally ascending or same
pops_all = [sum([True for point in points if abs(point[p]) > 0])/len(points)
            for p in range(len(labels))]
plt.figure()
plt.barh(range(len(labels)), pops_all,
        edgecolor = 'black', color = 'none', alpha = 0.5, height = 1, label = 'all clusters')
for i, cluster_choice in enumerate(cluster_choices):
    pops_choice = [sum([True for point in models_by_clusters[cluster_choice] if abs(point[p]) > 0])/len(models_by_clusters[cluster_choice])
               for p in range(len(labels))]
    plt.barh(range(len(labels)), pops_choice,
            color = colours[i], alpha = alphas[i], height = heights[i], label = 'cluster ' + str(cluster_choice))
plt.xlabel('presence')
plt.ylabel('parameter')
plt.yticks(range(len(labels)), labels_latex)
plt.legend()
plt.savefig(output_name + '_process_popularity' + '.svg', 
            dpi = 1000, bbox_inches='tight')

        
        
#%%        
# sample set of models (like one cluster), evaluate dynamics, find mean and standard deviation

# want for each observable numpy array
# dyynamics produces array in time
# stack multiple models vertically
# then can do mean, std over columns
# plot this vs target

# section settings:
# cluster_choices = [0, 1] # note: now taken from above section, enable if required separately
samples = 10000

# target data:
# note: datasets and observable labels must be encapsulated into lists
with open(target_data_pickle_file,
          'rb') as filestream:
    simulated_data = pickle.load(filestream)    
ts, sx, sy, sz = [simulated_data[x] for x in ['ts', 'sx', 'sy', 'sz']]
measurement_datasets = [sx, sy, sz]
measurement_observables = ['sigmax', 'sigmay', 'sigmaz']

# cumulative evaluated arrays combining all chosen clusters:
cumul_evaluated_arrays = {}

# turn each parameters vector into model and evaluate and save observables:
for j, chosen_cluster in enumerate(cluster_choices):
    model_set = models_by_clusters[chosen_cluster]
    vectors = random.sample(model_set, min(samples, len(model_set)))
    
    evaluation_ts = ts
    evaluated_datasets = {obs: [] for obs in measurement_observables} 
    model = learning_model.LearningModel()
    for vector in vectors:
        model.configure_to_params_vector((vector, labels, labels_latex),
                                         D = D,  
                                         qubit_initial_state = ops['plus'],
                                         defect_initial_state = ops['mm'])
        temp = model.calculate_dynamics(evaluation_ts, measurement_observables)
        for i in range(len(measurement_observables)):
            evaluated_datasets[measurement_observables[i]].append(temp[i])
        
    # stack sample models evaluated data:
    # entry for each observable is array, each row for one model,
    # each column for one evaluation time, to find mean and std along columns     
    evaluated_arrays = {obs: np.stack(evaluated_datasets[obs])
                        for obs in measurement_observables}
    
    means = {obs: evaluated_arrays[obs].mean(axis=0)
             for obs in measurement_observables}
    
    stds = {obs: evaluated_arrays[obs].std(axis=0)
             for obs in measurement_observables}
    
    for i, op in enumerate(measurement_observables):
        plt.figure()
        plt.xlabel('t (us)')
        plt.ylabel(ops_longlabels[op])
        plt.ylim([-1, 1])
        plt.plot(evaluation_ts, means[op], 'r-', linewidth = 0.7, alpha = 0.7)
        plt.fill_between(evaluation_ts, means[op]-stds[op], means[op]+stds[op],
                         alpha=0.4, color='tomato', label = 'cluster ' + str(chosen_cluster))
        plt.errorbar(ts, measurement_datasets[i], yerr = simulated_std,
                     fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target')
        plt.legend()
        plt.savefig(output_name + '_C' + str(chosen_cluster)
                    + '_sample' + str(min(samples, len(model_set)))
                    + '_' + op + '_comparison.svg', dpi = 1000, bbox_inches='tight')
     
        
    # also add to cumulative array:
    for obs in measurement_observables:
        if j == 0: # ie. first iteration
            cumul_evaluated_arrays[obs] = evaluated_arrays[obs]
        elif j > 0:
            cumul_evaluated_arrays[obs] = np.concatenate((cumul_evaluated_arrays[obs], evaluated_arrays[obs]), axis = 0)
    
# plot cumulative means and stds (combining all chosen clusters):
clusters_label = 'clusters ' + ''.join([str(x) + ', ' for x in cluster_choices])[:-2]
means = {obs: cumul_evaluated_arrays[obs].mean(axis=0)
         for obs in measurement_observables}
stds = {obs: cumul_evaluated_arrays[obs].std(axis=0)
         for obs in measurement_observables}
for i, op in enumerate(measurement_observables):
    plt.figure()
    plt.xlabel('t (us)')
    plt.ylabel(ops_longlabels[op])
    plt.ylim([-1, 1])
    plt.plot(evaluation_ts, means[op], 'r-', linewidth = 0.7, alpha = 0.7)
    plt.fill_between(evaluation_ts, means[op]-stds[op], means[op]+stds[op],
                     alpha=0.4, color='tomato', label = clusters_label)
    plt.errorbar(ts, measurement_datasets[i], yerr = simulated_std,
                 fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target')
    plt.legend()
    plt.savefig(output_name + '_C' + str(chosen_cluster)
                + '_sample' + str(min(samples, len(model_set)))
                + '_' + op + '_comparison.svg', dpi = 1000, bbox_inches='tight')
