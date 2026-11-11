#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)

Finds correlations for across selected clusters of models.
"""

# CHANGE: POSTERIORS NO LONGER SAVED - SWAP FOR IMPORTING LOG LIKELIHOODS PRIORS AND SUMMING 

import pandas as pd
import pickle
import seaborn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
#import copy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import multiprocessing

import configs
import learning_model
from definitions import observable_shorthand2pretty as ops_longlabels, ops

# global plotting settings (works locally):
fontsize = 6
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelpad'] = -0
plt.rcParams['axes.linewidth'] = 0.25
plt.rcParams['xtick.major.width'] = 0.25
plt.rcParams['ytick.major.width'] = 0.25
single_col_figsize=(3.41,2)
single_col_tall_figsize=(3.41,4)
single_col_square_figsize=(3.41,3.41)
colours = list(matplotlib.colormaps['Set1'].resampled(9)(range(9)))
colours += colours

# assuming same D for each og_file and same number of Rs for each combination
og_sources = ["Wit-Fig4-5-0_1", "Wit-Fig4-6-0_025", "Wit-Fig4-6-0_1", "Wit-Fig4-6-0_2", "Wit-Fig4-7-0_1"]
#og_sources = ["Wit-Fig4-6-0_025", "Wit-Fig4-7-0_1"]
Ds = [1,2,3]
og_sources_Ds = [(og_source, D) for og_source in og_sources for D in Ds]
human_labels = {"Wit-Fig4-5-0_1": '5\n0.1',
                "Wit-Fig4-6-0_025": '6\n0.025',
                "Wit-Fig4-6-0_1": '6\n0.1',
                "Wit-Fig4-6-0_2": '6\n0.2',
                "Wit-Fig4-7-0_1": '7\n0.1'}
    
    # restoring - damping (MHz)



def execute(og_source_D: tuple[str]) -> dict[tuple[list]]:
    
    og_source, D = og_source_D
    
    # settings:
    experiment_name = '261027-2M-exp'
    noise_stdev = 0 # set None if not included in file name
    Rs = [i for i in range(5)] # for D2
    config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
    if False: Rs_tag = ''.join([x + ',' for x in map(str, Rs)])[:-1] # lists all Rs - may crash if file name too long
    if len(Rs)>1: Rs_tag = str(Rs[0]) + 'thru' + str(Rs[-1]) 
    else: Rs_tag = str(Rs[0])
    clustering_name = 'e500'
    chosen_k = 4 # e1000: 0.1: 5, 0.05: 4/7, 0.01: 6 none good
    correlation_hierarchical_clustering_thresholds = [0.7, 0.5]
    
    # consequent setup:
    hyperparams = configs.get_hyperparams(config_name)
    if noise_stdev or True:
        experiment_name += '_std' + str(noise_stdev).replace('.','p')
    experiment_name += '_' + og_source
    output_name = (experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_'
                   + clustering_name + '_k' + str(chosen_k) + '_select')
    
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
    with open(outputs_each_k_file, 'rb') as filestream:
        outputs_each_k = pickle.load(filestream)
    assignments = list(outputs_each_k[chosen_k]['assignments'])
    with open(source_base + '_log_likelihoods_priors.pickle', 'rb') as filestream:
        log_likelihoods_priors = pickle.load(filestream)
    posteriors = [x[0] + x[1] for x in log_likelihoods_priors]
    
    
    # note: files needed for import given a choice of k are:
    # lowest R best model pickle (for extracting process labels)
    # combined_assignments pickle for that k
    # points (as subsampled and vectorised - prior to choice of k)
    # outputs for ks (contain assignments but also metrics for all k's - maybe redundant)
    # log likelihood priors pickle (independent of k again - maybe redundant)
    
    # !!! TO DO: check that posteriors use the correct temperature in calculation from log likelihood and prior
    
    
    #%%
    if True:
        # bar plot of clusters' popularity:
        plt.figure(figsize=single_col_figsize)
        clusters_popularity = [len(models_by_clusters[x]) for x in models_by_clusters]
        plt.bar(list(models_by_clusters.keys()), clusters_popularity,
                color = 'navy')
        plt.xlabel('cluster')
        plt.xticks(list(range(chosen_k)), labels = [str(x) for x in list(range(chosen_k))])
        plt.ylabel('number of models')
        plt.savefig(output_name + '_cluster_popularity' + '.svg', 
                    dpi = 1000, bbox_inches='tight')
        
        # ordering of clusters by popularity in DESCENDING order:
        clusters_by_pop_desc = sorted(range(len(clusters_popularity)), 
                                       key = lambda i: clusters_popularity[i], reverse=True)
        # note: no need to now order clusters labels by this
        # ordering and ordered set equivalent here because cluster labels are a range(0, chosen_k)
    
            
    
    #%%
    # find correlations, dendrograms, and clustered parameters in selected cluster combinations
    # for now: do doing 1) all and 2) most popular cluster
    
    if True:
        cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
        
        # list of lists, each inner list for combinations of cluster to analyse together: 
        cluster_combinations = []
        # each separately - CURRENTLY DISABLED:
        if False:
            cluster_combinations.extend([x] for x in (models_by_clusters.keys()))
        # all together: 
        cluster_combinations.append(list(list(models_by_clusters.keys())))
        # most populare cluster:
        cluster_combinations.append(clusters_by_pop_desc[0:1])
         
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
            # note: assuming only one system, replace with nicer labels:
            labels_cmat = [x.replace('S1','S') for x in labels_latex]
            for i in range(len(labels_cmat)):
                labels_cmat[i] = labels_cmat[i].replace(',', ' ')#'$\\leftrightarrow$')
                for j in range(10):
                    labels_cmat[i] = labels_cmat[i].replace('V' + str(j), 'V' + '$_' + str(j) + '$')
            data = data.rename(columns = {labels[i]: labels_cmat[i] for i in range(len(labels))})
            CM = data.corr()
            dissimilarity = 1 - abs(CM)
            Z = linkage(squareform(dissimilarity), 'complete')
            
            # plot correlation matrix:
            plt.figure(figsize=single_col_square_figsize)
            seaborn.heatmap(CM, annot=False, cmap=cmap, fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
            # colormaps: coolwarm, PiYG
            plt.savefig(output_name + '_Cs' + Cs_tag + '_mat.svg', dpi = 1000, bbox_inches='tight')
            
            # hierarchical clustering:
            
            # plot dendrogram: 
            with plt.rc_context({'lines.linewidth': 0.5}):
                plt.figure(figsize=single_col_square_figsize)
                #plt.gca().tick_params(axis='y', rotation=0)
                dendrogram(Z, labels=data.columns, orientation='right'
                           ,leaf_rotation=0
                           ,leaf_font_size=fontsize
                           )
                plt.ylabel('dissimilarity')
                plt.savefig(output_name + '_Cs' + Cs_tag + '_dendrogram.svg', dpi = 1000, bbox_inches='tight')
            
            # plot correlation matrix with clustered parameters
            # note: now clustering parameters not models!!
            if True:
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
                    plt.figure(figsize=single_col_square_figsize)
                    clust_CM = clustered.corr()
                    # seaborn.heatmap(round(clust_CM,2), cmap='RdBu', annot=True, annot_kws={"size": 7}, vmin=-1, vmax=1);
                    seaborn.heatmap(clust_CM, cmap=cmap, annot=False, vmin=-1, vmax=1);
                    plt.savefig(output_name + '_Cs' + Cs_tag + '_thr' + str(threshold) + '_cmat.svg', dpi = 1000, bbox_inches='tight')
                
            
        
    # %%
    # popularity of different processes:
    # CURRENTLY compare UP TO 5 most popular clusters (slicing capped by number of clusters)
    if not True:
        cluster_choices = clusters_by_pop_desc[0:5]
        # cluster_choices = [2,3,4] # for LN D=1
        # formatting - must have options for at least each cluster choice, can be longer:
        
        heights = [0.8, 0.65, 0.40, 0.25, 0.1] # ideally descending
        alphas = [0.3, 0.3, 0.5, 0.7, 1] # ideally ascending or same
        pops_all = [sum([True for point in points if abs(point[p]) > 0])/len(points)
                    for p in range(len(labels))]
        plt.figure(figsize=single_col_tall_figsize)
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
        plt.legend(fontsize = 'small')
        plt.savefig(output_name + '_process_popularity' + '.svg', 
                    dpi = 1000, bbox_inches='tight')
    
            
            
    #%%        
    # sample subset of models from each of cluster choices, evaluate dynamics, find mean and standard deviation:
    
    # want for each observable numpy array
    # dyynamics produces array in time
    # stack multiple models vertically
    # then can do mean, std over columns
    # plot this vs target
    
    # section settings:
    cluster_choices = list(range(chosen_k)) # note: now taken from above section, enable if required separately
    samples = 1000
    
    # target data:
    # note: datasets and observable labels must be encapsulated into lists
    imported_data = np.genfromtxt(og_source + '.csv', delimiter=',').transpose()
    ts, sx = imported_data[0]/1000, imported_data[1] # note: convert time to us here (from original ns)
    measurement_datasets = [sx]
    measurement_observables = ['sigmax']
    additional_observables = ['sigmay']
    evaluation_observables = measurement_observables + additional_observables
    
    # cumulative evaluated arrays combining all chosen clusters,
    # also means and stds now as dictionaries storing this for each chosen cluster for overlaying:
    cumul_evaluated_arrays = {}
    means = {}
    stds = {}
    
    
    # turn each parameters vector into model and evaluate and save observables:
    for j, chosen_cluster in enumerate(cluster_choices):
        model_set = models_by_clusters[chosen_cluster]
        vectors = random.sample(model_set, min(samples, len(model_set)))
        
        evaluation_ts = ts
        evaluated_datasets = {obs: [] for obs in evaluation_observables} 
        model = learning_model.LearningModel()
        for vector in vectors:
            model.configure_to_params_vector((vector, labels, labels_latex),
                                             D = D,  
                                             qubit_initial_state = ops['plus'],
                                             defect_initial_state = ops['mm'])
            temp = model.calculate_dynamics(evaluation_ts, evaluation_observables)
            for i in range(len(evaluation_observables)):
                evaluated_datasets[evaluation_observables[i]].append(temp[i])
            
        # stack sample models evaluated data:
        # entry for each observable is array, each row for one model,
        # each column for one evaluation time, to find mean and std along columns     
        evaluated_arrays = {obs: np.stack(evaluated_datasets[obs])
                            for obs in evaluation_observables}
        
        means[chosen_cluster] = {obs: evaluated_arrays[obs].mean(axis=0)
                 for obs in evaluation_observables}
        
        stds[chosen_cluster] = {obs: evaluated_arrays[obs].std(axis=0)
                 for obs in evaluation_observables}
        
        if not True: # individual plots for each cluster
            for i, op in enumerate(evaluation_observables):
                plt.figure(figsize=single_col_figsize)
                plt.xlabel(r't ($\mu s$)')
                plt.ylabel(ops_longlabels[op])
                plt.ylim([-1, 1])
                plt.plot(evaluation_ts, means[chosen_cluster][op], 'r-', linewidth = 0.7, alpha = 0.7)
                plt.fill_between(evaluation_ts, means[chosen_cluster][op]-stds[chosen_cluster][op], means[chosen_cluster][op]+stds[chosen_cluster][op],
                                 alpha=0.3, color='tomato', label = 'cluster ' + str(chosen_cluster))
                plt.errorbar(ts, measurement_datasets[i], yerr = noise_stdev,
                             fmt = 'b.', ecolor = 'b', markersize = 1, label = 'target', alpha = 1, linewidth = 0.5)
                plt.legend(fontsize = 'small')
                plt.savefig(output_name + '_C' + str(chosen_cluster)
                            + '_samp' + str(min(samples, len(model_set)))
                            + '_' + op + '_solo.svg', dpi = 1000, bbox_inches='tight')
             
            
        # also add to cumulative array:
        for obs in evaluation_observables:
            if j == 0: # ie. first iteration
                cumul_evaluated_arrays[obs] = evaluated_arrays[obs]
            elif j > 0:
                cumul_evaluated_arrays[obs] = np.concatenate((cumul_evaluated_arrays[obs], evaluated_arrays[obs]), axis = 0)
        
    #%%    
    # plot dynamics sweep of all chosen clusters combined, and all chosen clusters separate dynamics overlay 
    
    cumul_dynamics = {} # one output dictionary of dynamics for each combination of og_file and D,
    # where keys are evaluation observables and entries are (means, stds) 
    
    if True:
        cluster_choices = list(range(chosen_k))
        if False: cluster_choices = [4,5] # only showing dynamics of handpicked clusters
        
        key = tuple(cluster_choices)
        clusters_label = 'clusters ' + ''.join([str(x) + ', ' for x in cluster_choices])[:-2]
        clusters_label_short = ''.join([str(x) + ',' for x in cluster_choices])[:-1]
        means[key] = {obs: cumul_evaluated_arrays[obs].mean(axis=0)
                 for obs in evaluation_observables}
        stds[key] = {obs: cumul_evaluated_arrays[obs].std(axis=0)
                 for obs in evaluation_observables}
        # note: this computes moments across clusters, earlier moments are calculated individually for each cluster 
        
        curveLW = 0.75
        ebarLW = 0.35
        errorbarformatting = {'linestyle': '', 
                              'linewidth': 0, # this would connect the points - not wanted
                              #'ecolor': 'k', 'elinewidth': ebarLW, 'capsize': 0.35, 'capthick': ebarLW,
                              'marker': 'D' , 'mfc': 'k', 'mec': 'none', 'mew': ebarLW, 'markersize': 1.0, 
                              'alpha': 0.5,
                              'label': 'target'
                              }
        
        for i, op in enumerate(evaluation_observables):
            
            # plot cumulative means and stds (combining all chosen clusters):
            key = tuple(cluster_choices)
            plt.figure(figsize=single_col_figsize)
            plt.xlabel(r't ($\mu s$)')
            plt.ylabel(ops_longlabels[op])
            plt.ylim([-0.2, 0.8])
            plt.plot(evaluation_ts, means[key][op], 'r-', linewidth = curveLW, alpha = 0.75)
            plt.fill_between(evaluation_ts, means[key][op]-stds[key][op], means[key][op]+stds[key][op],
                             alpha=0.2, facecolor='tomato', edgecolor = 'none', label = clusters_label)
            if op in measurement_observables: # assuming measurement observables not repeated in additional observables (which they shouldn't be)
                plt.plot(ts, measurement_datasets[i], **errorbarformatting)
                target_datapoints = measurement_datasets[i]
            else: target_datapoints = [None for x in measurement_datasets[0]]
            #plt.legend(fontsize = 'small')
            plt.savefig(output_name + '_Cs' + clusters_label_short
                        + '_samp' + str(min(samples, len(model_set)))
                        + '_' + op + '_lump_adhoc.svg', dpi = 1000, bbox_inches='tight')
            # # save cumulative dynamics (deepcopy should not be necessary) as output to overlay different datasets/Ds
            cumul_dynamics[op] = (ts, means[key][op], stds[key][op], target_datapoints) # tuple of lists
            
            
            # plot together means with stds filling for different clusters (overlay)
            # note: maybe manually specify list of colours
            plt.figure(figsize=single_col_figsize)
            plt.xlabel(r't ($\mu s$)')
            plt.ylabel(ops_longlabels[op])
            plt.ylim([-0.2, 0.8])
            for j, key in enumerate(cluster_choices): 
                plt.plot(evaluation_ts, means[key][op], '-', linewidth = curveLW, alpha = 0.75, color = colours[j])
                plt.fill_between(evaluation_ts, means[key][op]-stds[key][op], means[key][op]+stds[key][op],
                                 alpha=0.1, label = '' + str(key), facecolor = colours[j], edgecolor = 'none')
            if op in measurement_observables: # assuming measurement observables not repeated in additional observables (which they shouldn't be)
                plt.plot(ts, measurement_datasets[i], **errorbarformatting)
            plt.legend(ncol = 2, fontsize = 'small')
            plt.savefig(output_name + '_Cs' + clusters_label_short
                        + '_samp' + str(min(samples, len(model_set)))
                        + '_' + op + '_overlay_adhoc.svg', dpi = 1000, bbox_inches='tight')
            
            
    
    
    
    #%%
    # also plot champion parameter vectors for chosen clusters:
    if True:
        champions = {}
        #champ_posteriors = {}
        
        for c in cluster_choices:
        
            # find highest posterior and corresponding parameter vector (point) for each cluster assignment c
            champion, posterior, assignment = max(filter(lambda x: x[2] == c, zip(points, posteriors, assignments)),
                                                key = lambda x: x[1])
            champions[assignment] = champion
            #champ_posteriors[assignment] = posterior
            #vectorised_champion = (champion, labels, labels_latex)
            
        champions_array = np.stack([champions[x] for x in cluster_choices])
        
        plt.figure(figsize=single_col_tall_figsize)
        axisfontsize = fontsize
        img = plt.imshow(np.transpose(champions_array), interpolation='none', aspect='1')
        cbar = plt.colorbar(img, fraction=0.015) # , cmap='viridis' ???? not doing anything?
        plt.set_cmap('viridis')
        cbar.ax.tick_params(labelsize=fontsize)
        plt.xticks(ticks = [x for x in range(champions_array.shape[0])], labels = [x for x in range(champions_array.shape[0])])
        plt.ylabel('parameter', fontsize=axisfontsize)
        plt.yticks(range(len(labels_latex)), labels = labels_latex)
        plt.xlabel('cluster', fontsize=axisfontsize)
        plt.gca().tick_params(axis='both', which='major', labelsize=4)
        plt.savefig(output_name + '_champions_k' + str(chosen_k) + '.svg',  dpi = 1000, bbox_inches='tight')
        plt.savefig(output_name + '_champions_k' + str(chosen_k) + '.png',  dpi = 1000, bbox_inches='tight')
    
        
    
    #%%
    # histogram of chosen parameter values as a marginal of posterior distrubution (pdf)
    if False:
        points_array = np.stack(points) # columns for each parameter so that this[:, j] are j-th parameter values
        
        for j in range(20):
            vals = points_array[:, j]
            if True: 
                # remove zeros:
                vals_onlynonzero = vals[vals != 0]
            param_label = labels_latex[j]
            filename_label = labels[j]
            plt.figure(figsize=single_col_figsize)
            #bins = 10 ** np.linspace(min(vals), max(vals), 100)
            if False:
                plt.hist(vals, alpha = 0.7, color = 'turquoise'
                         ,bins = 100, label = 'all'
                         )
            plt.hist(vals_onlynonzero, alpha = 0.5, color = 'firebrick'
                     ,bins = 400, label = 'non-zero'
                     )
            plt.xlabel(param_label)
            plt.ylim(0,600)
            #plt.xscale('log')
            plt.ylabel('count')
            
            plt.savefig(output_name
                        + '_samp' + str(min(samples, len(model_set)))
                        + '_hist' + filename_label + '.svg', dpi = 1000, bbox_inches='tight')
    
    #%%
    # 2D histogram of ad-hoc chosen parameters (marginals now)
    
    if False:
        points_array = np.stack(points) # columns for each parameter so that this[:, j] are j-th parameter values
        
        i, j = 2, 3
        vals_first = points_array[:, i]
        param_label_first = labels_latex[i]
        filename_label_first = labels[i]
        vals_second = points_array[:, j]
        param_label_second = labels_latex[j]
        filename_label_second = labels[j]
        if False: # can't remove zeros or not same length (no longer coordinates of same vectors)
            vals_first = vals_first[vals_first != 0]
            vals_second = vals_second[vals_second != 0]
        plt.figure(figsize=single_col_figsize)
        cs, xs, ys, _ = plt.hist2d(vals_first, vals_second, bins=100, density = False, cmin=1, cmax=1000)
        # temp = plt.hist2d(vals_first, vals_second, bins=50, density = False, cmap = 'Purples',
        #                    cmin = 1, cmax = max(temp[0][0])/100)
        
        plt.xlabel(param_label_first)
        #plt.ylim(0,600)
        #plt.xscale('log')
        plt.ylabel(param_label_second)
        plt.gca().set_aspect('equal')
        filename_label = filename_label_first + '-' + filename_label_second
        
        plt.savefig(output_name
                    + '_samp' + str(min(samples, len(model_set)))
                    + '_hist' + filename_label + '_adhoc.svg', dpi = 1000, bbox_inches='tight')
        
        # i, j = 2, 4
        # vals_first = points_array[:, i]
        # param_label_first = labels_latex[i]
        # filename_label_first = labels[i]
        # vals_second = points_array[:, j]
        # param_label_second = labels_latex[j]
        # filename_label_second = labels[j]
        # if False: # can't remove zeros or not same length (no longer coordinates of same vectors)
        #     vals_first = vals_first[vals_first != 0]
        #     vals_second = vals_second[vals_second != 0]
        # plt.figure(figsize=single_col_figsize)
        # plt.hist2d(vals_first, vals_second, bins=50, density = True, cmin=0, cmap = 'Purples')
        # plt.xlabel(param_label_first)
        # #plt.ylim(0,600)
        # #plt.xscale('log')
        # plt.ylabel(param_label_second)
        # plt.gca().set_aspect('equal')
        # filename_label = filename_label_first + '-' + filename_label_second
        
        # plt.savefig(output_name
        #             + '_samp' + str(min(samples, len(model_set)))
        #             + '_hist' + filename_label + '_adhoc.svg', dpi = 1000, bbox_inches='tight')
         

    #%%
    return cumul_dynamics                         

if __name__ == '__main__' and True:
        

    with multiprocessing.Pool() as pool:
        
        cumul_dynamics_sets = pool.map(execute, og_sources_Ds)

#%%
# cumul dynamics set: list of dicts (keys obs, entries tuples of (ts, means, stds, target))
# corresponds to elements of list (og_source, D)
experiment_name = '261027-2M-exp'
clustering_name = 'e500'
chosen_k = 5
filename = experiment_name + '_' + clustering_name + '_k' + str(chosen_k)
curveLW = 0.75
ebarLW = 0.5
errorbarformatting = {}

plotting_Ds = Ds

# plot for chosen plotting Ds, all evaluation observables (assumed same for all combinations of og_sources and Ds):
# ...the overlay of dynamics for different og_sources (traces)
for plotting_D in plotting_Ds:
    for op in cumul_dynamics_sets[0].keys():
        plt.figure(figsize=single_col_figsize)
        plt.ylabel(ops_longlabels[op])
        plt.xlabel(r't ($\mu s$)')
        for i, og_source_D in enumerate(og_sources_Ds): 
            # i is the index of the og_file (trace) to plot for this observable and D
            og_source, D = og_source_D
            if D == plotting_D:
                cumul_dynamics_set = cumul_dynamics_sets[i]
                ts, means, stds, target = cumul_dynamics_set[op]
                plt.plot(ts, means, '-', linewidth = curveLW, alpha = 0.75, label = human_labels[og_source], color=colours[i])
                plt.fill_between(ts, means-stds, means+stds,
                                 alpha=0.2, facecolor=colours[i], edgecolor = 'none')
                plt.plot(ts, target, linestyle = '',
                         linewidth = 0, # this would connect the points - not wanted
                         marker = '.' , mfc = 'none', mec = colours[i], mew = ebarLW, markersize = 4.0, 
                         alpha = 1,
                         #label = 'target'
                         )
        plt.legend(title = r'$D = $' + str(plotting_D) + '\n' + r'$rest.$' + ' (MHz)\n' + r'$damp.$' + ' (MHz)' )
        plt.savefig(filename + '_' + str(op) + '_' + str(plotting_D)
                    + '.svg', dpi = 1000, bbox_inches='tight')
    