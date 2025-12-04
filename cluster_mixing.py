#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import sys # for passing command line arguments
import json
import sklearn.cluster
import sklearn.metrics
import kneed
import pandas as pd #
import pickle
import configs
import seaborn #
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np
import time
import copy

# settings and source data: # '250818-sim-1T-4JL-2tweak' is nice fit
experiment_name = '251128-smallnoise' + '_Wit-Fig4-6-0_025' # including experiment base and source file name
#experiment_name = '251110-100k' + '_Wit-Fig4-6-0_025' # including experiment base and source file name

config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'

D = 2
Rs = [2, 5, 8, 10, 11, 12, 13, 16, 17, 22, 23, 26, 27, 29, 30] # for combining chains
Rs_tag = ''.join([x + ',' for x in map(str, Rs)])[:-1]
hyperparams = configs.get_hyperparams(config_name)
output_name = experiment_name + '_' + config_name + '_D' + str(D) + '_Rs' + Rs_tag + '_clustering-every100'
min_clusters = 2
max_clusters = 10
loss_threshold = False # if zero, watch the conditional 
bounds = []
verbosity = 0
burn = 0
subsample = 100 # take every however-many-eth point; 1 means every point taken
only_take_annealed = True

# clustering choice (by Liouvllians or parameter vectors)
# vectorisation = 'Liouvillian'
vectorisation = 'parameters'
if vectorisation == 'parameters':
    #config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
    hyperparams = configs.get_hyperparams(config_name)

 
# container for points to cluster (vectorised and decomplexified Liouvillians, or parameter vectors),
# as well as corresponding loss and posterior
points = []
losses, posteriors = [], []

# time trackers for profiling:
new_time = time.time()
time_last = new_time


# number subsampled clustered models taken from each chain:
taken_from_each_R_subsampled = []

labels_obtained = False
for R in Rs:
    
    # import accepted loss values and accepted proposals from same output dictionary:
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R)
    with open(filename + '_proposals.pickle',
              'rb') as filestream:
        proposals = pickle.load(filestream)
    accepted_annealing_flags = [x for (x, y) in zip(proposals['annealed'], proposals['acceptance']) if y]
    accepted_proposals = [x for (x,y) in zip(proposals['proposals'], accepted_annealing_flags)
                          if (y or not only_take_annealed)]
    accepted_losses = [x for (x,y,z) in zip(proposals['loss'][1:], proposals['acceptance'], proposals['annealed'])
                       if y and (z or not only_take_annealed)]
    accepted_posteriors = [x for (x,y,z) in zip(proposals['log_posterior'][1:], proposals['acceptance'], proposals['annealed'])
                       if y and (z or not only_take_annealed)]
    # !!! note: only accepted proposals are saved in proposals, 
    # whereas other entries in proposals dictionary are for all proposals regardless of acceptance
    
    # get parameter labels off of 1st proposal:
    if not labels_obtained:
        hyperparams = configs.get_hyperparams(config_name)
        _, labels, labels_latex = accepted_proposals[0].vectorise_under_library(hyperparameters = hyperparams)
        
    
    # collect all points including loss and posterior:
    
    # split into segments determined by bounds:    
    bounds = [
              (0, len(accepted_losses))
              ]
    if len(bounds) > 1: # plot chain segments determined by bounds:
        indices = list(range(len(accepted_losses)))
        plt.figure()
        plt.plot(indices, accepted_losses, '-', c = 'orange', linewidth = 0.5)
        plt.yscale('log')
        ymin = min(accepted_losses)
        ymax = max(accepted_losses)
        plt.ylabel('loss')
        plt.xlabel('accepted model')
        for region in bounds:
            plt.plot([region[0], region[0]], [ymin, ymax], 'r-', linewidth = 1)
            plt.plot([region[1], region[1]], [ymin, ymax], 'g-', linewidth = 1)
        plt.savefig(output_name + '_chain_segments.svg')
        plt.clf()
        
    # remove points with loss below some threshold - don't combine this with bounds!
    elif type(loss_threshold) in [int, float]:
        print('Earlier: ' + str(len(accepted_proposals)), flush = True)
        working_proposals = [x for (x, y) in zip(accepted_proposals, accepted_losses) if y < loss_threshold]
        print('After: ' + str(len(accepted_proposals)), flush = True)
    
    # take only points between the specified regions (sets of bounds),
    # also corresponding losses and posteriors:
    if True:
        working_proposals = []
        working_losses, working_posteriors = [], []
        for region in bounds:
            working_proposals.extend(accepted_proposals[region[0]:region[1]])
            working_losses.extend(accepted_losses[region[0]:region[1]])
            working_posteriors.extend(accepted_posteriors[region[0]:region[1]])
    new_points = []
    
    # turn proposals into points (vectors):
    if vectorisation == 'Liouvillian': # use Liouvillian
        for new_model in working_proposals:
            # build Liouvillian, turn into 1D vector, separate real and imaginary parts and concatenate:
            # note: scikit-learn cannot work with complex vectors
            Liouvillian_mat = new_model.build_Liouvillian()
            Liouvillian_vect_complex = Liouvillian_mat.ravel()
            Liouvillian_vect_separated = np.concatenate((Liouvillian_vect_complex.real, Liouvillian_vect_complex.imag))
            new_points.append(Liouvillian_vect_separated)
    elif vectorisation == 'parameters': # use model vector
        for new_model in working_proposals:
            new_points.append(new_model.vectorise_under_library(hyperparameters = hyperparams)[0])
    
    taken_from_each_R_subsampled.append(len(working_proposals[0::subsample]))
    points.extend(new_points[0::subsample])
    losses.extend(working_losses[0::subsample])
    posteriors.extend(working_posteriors[0::subsample])
    
    
# final array to feed into clusterer 
# note: each row a different model:
points_array = np.stack(points)
#points_array = points_array[0::subsample,:] # if sampling subsampling combined chains, not now - changes edge cases!


# also export lists of points, losses, posteriors:
with open(output_name + '_points.pickle', 'wb') as filestream:
    pickle.dump(points, filestream)
with open(output_name + '_losses.pickle', 'wb') as filestream:
    pickle.dump(losses, filestream)
with open(output_name + '_posteriors.pickle', 'wb') as filestream:
    pickle.dump(posteriors, filestream)


    
print('\n.....\ndata preparation time pre-clustering (s):' 
      + str(np.round((new_time := time.time()) - time_last,2)) + '\n.....\n', flush = True)
  

#%% clustering execution:

clusters_counts = list(range(min_clusters, max_clusters + 1))

# containers for clustering outputs:
SSEs = []
centres = []
assignments = []
iters_required = []
silhouette_scores = [] # higher is better

# go over all cluster numbers:
time_last = time.time()
outputs_each_k = {}
for k in clusters_counts:

    # perform k-means clustering:
    kmeans = sklearn.cluster.KMeans(
        init = "random", # "random" or "k-means++"
        n_clusters = k, # number of clusters
        n_init = 10, # number of random initialisations
        max_iter = 30, # maximum centroid adjustments 
        # note: usually converges in just a handful of iterations
        random_state = None # just leave this
        , verbose = verbosity
        )
    kmeans.fit(points_array)
    
    # save outputs for latest k:
    # note: rewritten when looping over multiple k's
    SSEs.append(kmeans.inertia_)
    centres.append(kmeans.cluster_centers_)
    assignments.append(kmeans.labels_)
    iters_required.append(kmeans.n_iter_)
    silhouette_scores.append(sklearn.metrics.silhouette_score(points_array, kmeans.labels_))
    
    # save outputs for this k:
    outputs_each_k[k] = {}    
    outputs_each_k[k]['SSEs'] = kmeans.inertia_
    outputs_each_k[k]['centres'] = kmeans.cluster_centers_
    outputs_each_k[k]['assignments'] = kmeans.labels_
    outputs_each_k[k]['iters_required'] = kmeans.n_iter_
    outputs_each_k[k]['silhouette_scores'] = sklearn.metrics.silhouette_score(points_array, kmeans.labels_)

    # profiling:
    if verbosity > 0 or True:
        print('\n.....\ndone k = ' + str(k) + '\nthis k time (s):' + str(np.round((new_time := time.time()) - time_last,2)) 
              + '\n.....\n', flush = True)
        time_last = new_time

# export dictionary of all outfuts for each k (includes some outputs also saved in other files)
# variable number of coordinates so variable dimension array as each element of list, hence pickle:
with open(output_name + '_outputs_each_k.pickle', 'wb') as filestream:
    pickle.dump(outputs_each_k, filestream)



#%% metrics to find k:

# automatically find knee/elbow aka maximum curvature point:
knee_finder = kneed.KneeLocator(clusters_counts,
                                SSEs,
                                curve="convex",
                                direction="decreasing"
                                )
elbow = knee_finder.elbow
print('\nElbow found at ' + str(elbow) + ' clusters.\n', flush = True)

# # save dictionary containing successfully imported filenames, explored cluster numbers, 
# # and list of assignments (in order of filenames) for each explored cluster number:
# clustering_output = {'filenames': filenames,
#                      #'clusters_counts': clusters_counts, # already available in keys of assignments
#                      'assignments': {int(clusters_counts[i]): [int(x) for x in assignments[i]] for i in range(len(clusters_counts))}
#                      }
# with open(output_name + '_clustering_output.json', 'w') as filestream:
#     json.dump(clustering_output,  filestream)

# save SSE and silhouette score vs number of clusters as csv files:
np.savetxt(output_name + '_clustering_SSEs.csv', 
           np.transpose([clusters_counts, SSEs]),
           header = 'clusters,SSEs',
           delimiter = ',', comments = '')
np.savetxt(output_name + '_clustering_silhouette_scores.csv', 
           np.transpose([clusters_counts, silhouette_scores]),
           header = 'clusters,silhouettes',
           delimiter = ',', comments = '')

# xlabels for numbers of clusters with just edges (to avoid overlapping when dense)
x_tick_labels = [clusters_counts[0]] + ['' for x in range(len(clusters_counts)-2)] + [clusters_counts[-1]]
if not (type(elbow) == type(None)):
    x_tick_labels[elbow - 
    clusters_counts[0]] = elbow

# plot and save SSE vs number of clusters:
plt.figure(tight_layout = True)
plt.plot(clusters_counts, SSEs, 'b')
plt.xlabel('number of clusters')
plt.ylabel('SSE')
plt.xticks(ticks = clusters_counts, labels = x_tick_labels)
plt.title('elbow found at ' + str(elbow))
plt.savefig(output_name + '_clustering_SSEs.svg',  dpi = 1000, bbox_inches='tight')
plt.clf()

# plot and save silhouette score vs number of clusters
plt.figure(tight_layout = True)
plt.plot(clusters_counts, silhouette_scores, 'r')
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.ylim([-0.1,1])
plt.xticks(clusters_counts, x_tick_labels)
#plt.title(output_name)
plt.savefig(output_name + '_clustering_silhouette_scores.svg',  dpi = 1000, bbox_inches='tight')
plt.clf()
           

#%% cluster assignments big file:

# save cluster assignment for each point and also cluster centres (latter probably not very useful as it's the Liouvillian coordinates)
# each for different k, then each column is a different point
np.savetxt(output_name + '_clustering_assignments.csv',
           assignments,
           header = 'assignments',
           delimiter = ',', comments = '')
# variable number of coordinates so variable dimension array as each element of list, hence pickle:
with open(output_name + '_clustering_centres.pickle', 'wb') as filestream:
    pickle.dump(centres, filestream)
    
print('Finished exporting metrics', flush = True)


del proposals # this MAY speed things up before reallocation below - UNTESTED



#%% collate clustered models:
# create dictionary where keys are cluster labels,
# and entries are lists of all parameter vectors for that cluster;
# whole dictionary specific to k - use elbow for now 

# go over selected ks:
ks = clusters_counts
for k in ks:
    
    # make dictionary where keys are cluster labels and entries all points (parameter vectors) in that cluster:
    points_by_clusters = {}
    for label in set(outputs_each_k[k]['assignments']): # label (number) for each cluster
        points_by_clusters[label] = [x for (x,y) in zip(points, outputs_each_k[k]['assignments'])
                                     if y == label]
    
    # save as pickle:
    with open(output_name + '_k' + str(k) + '_combined_assignments.pickle', 'wb') as filestream:
        pickle.dump(points_by_clusters, filestream)
    

    
#%% plot assignments and centres given with k = elbow

if not False:
    
    ks = clusters_counts # ks to save assignment and centres for - can be cluster_counts, [elbow], or other
    # ks = [elbow]
    
    for k in ks:
    
        final_centres = outputs_each_k[k]['centres']
        final_assignments = outputs_each_k[k]['assignments']
        # indices in combined list marking where each chain begins:
        aux = [0] + [sum(taken_from_each_R_subsampled[:i+1]) for i in range(len(taken_from_each_R_subsampled))]
        
        
        # once again get accepted losses to plot against assignment:
        for i, R in enumerate(Rs):
        
            # import accepted loss values and accepted proposals from same output dictionary:
            # note: already imported earlier for clustering, but not all retained in memory,
            # hence must be imported again tor these plots
            filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R)
            with open(filename + '_proposals.pickle',
                      'rb') as filestream:
                proposals = pickle.load(filestream)
            accepted_proposals = [x for (x,y) in zip(proposals['proposals'], accepted_annealing_flags)
                                  if (y or not only_take_annealed)]
            accepted_losses = [x for (x,y,z) in zip(proposals['loss'][1:], proposals['acceptance'], proposals['annealed'])
                               if y and (z or not only_take_annealed)]
            
            # assignment curve (integer values marking pertinent cluster with -1 for burn)
            overlay = [-1 for x in range(burn)] + [x for x in final_assignments[aux[i]:aux[i+1]]]
            
            # subsample also indices and accepted losses for plotting 
            indices = list(range(len(accepted_losses)))
            indices_overlay = indices[0::subsample]
            
            # plot assignments for this chain
            fig, ax1 = plt.subplots(tight_layout = True)
            ax1.plot(indices, accepted_losses, c = 'orange')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('loss', c='orange')
            ax1.set_yscale('log')
            ax2 = ax1.twinx()
            ax2.plot(indices_overlay, overlay, ' ', marker='_', c='blue', alpha = 0.8)
            ax2.set_ylim([-0.2, k+0.2])
            ax2.set_ylabel('assignment', c='blue')
            ax2.set_yticks(list(range(k)))
            #plt.xticks(clusters_counts, x_tick_labels)
            #ax1.set_title('assignment to clusters')
            fig.savefig(output_name + '_R' + str(R) + '_assignment_k' + str(k) + '.svg',  dpi = 1000, bbox_inches='tight')
            plt.cla()
            
            # this is already available for the all-k output saved above so currently disabled
            if False:
                np.savetxt(output_name + '_R' + str(R)  + '_assignment_k' + str(k) + '.csv',
                           overlay,
                           header = 'assignment',
                           delimiter = ',', comments = '')
                
        # plot and save cluster centres as parameter vectors (rows for each parameter, columns for each cluster):
        np.savetxt(output_name + '_centres_k' + str(k) + '.csv',
                   final_centres,
                   #header = 'cluster centre parameters vector',
                   delimiter = ',', comments = '')
        plt.figure()
        axisfontsize = 6
        img = plt.imshow(np.transpose(final_centres), interpolation='none', aspect='1')
        cbar = plt.colorbar(img, fraction=0.015) # , cmap='viridis' ???? not doing anything?
        plt.set_cmap('viridis')
        cbar.ax.tick_params(labelsize=6)
        plt.xticks(ticks = [x for x in range(final_centres.shape[0])], labels = [x for x in range(final_centres.shape[0])])
        plt.ylabel('parameter', fontsize=axisfontsize)
        plt.yticks(range(len(labels_latex)), labels = labels_latex)
        plt.xlabel('cluster', fontsize=axisfontsize)
        plt.gca().tick_params(axis='both', which='major', labelsize=4)
        plt.savefig(output_name + '_centres_k' + str(k) + '.svg',  dpi = 1000, bbox_inches='tight')
        plt.savefig(output_name + '_centres_k' + str(k) + '.png',  dpi = 1000, bbox_inches='tight')
        plt.clf()
        # note: viewing the svg in ubuntu's image viewer interpolates between the blocks
        # - this is not a problem with the file but with the viewer
        
        
        print('Finished potting assignments of consituent models for each chain', flush = True)

        
        

#%% for testing:

# clusters_counts = [2]
# import pickle
# import matplotlib.pyplot as plt
# import configs
# import numpy as np

# config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
# output_name = 'test'
# hyperparams = configs.get_hyperparams(config_name)
# with open('251122-run_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle'
#           , 'rb') as filestream:
#     dummy = pickle.load(filestream)
# _, labels, labels_latex = dummy.vectorise_under_library(hyperparameters = hyperparams)
# with open('251121-annealed_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_Rs1_clustering-test_outputs_each_k.pickle'
#           , 'rb') as filestream:
#     outputs_each_k = pickle.load(filestream)
# k = 2
# final_centres = outputs_each_k[k]['centres']


# plt.figure()
# axisfontsize = 6
# img = plt.imshow(np.transpose(final_centres), interpolation='none', aspect='1')
# cbar = plt.colorbar(img, fraction=0.015) # , cmap='viridis' ???? not doing anything?
# plt.set_cmap('viridis')
# cbar.ax.tick_params(labelsize=6)
# plt.xticks(ticks = [x for x in range(final_centres.shape[0])], labels = [x for x in range(final_centres.shape[0])])
# #plt.yticks(ticks = [x for x in range(final_centres.shape[1])])#, labels = [x for x in range(final_centres.shape[1])])
# plt.ylabel('parameter', fontsize=axisfontsize)
# plt.yticks(range(len(labels_latex)), labels = labels_latex)
# plt.xlabel('cluster', fontsize=axisfontsize)
# plt.gca().tick_params(axis='both', which='major', labelsize=4)
# #plt.savefig(output_name + '_centres_k' + str(k) + '.svg',  dpi = 1000, bbox_inches='tight')
# plt.savefig(output_name + '_centres_k' + str(k) + '.png',  dpi = 1000, bbox_inches='tight')
# plt.clf()
