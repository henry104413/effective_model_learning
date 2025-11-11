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

# settings and source data: # '250818-sim-1T-4JL-2tweak' is nice fit
experiment_name = '251110-1M-narrower' + '_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 2
Rs = [2]
hyperparams = configs.get_hyperparams(config_name)
output_name = experiment_name + '_clustering_mixing_test_1'
min_clusters = 2
max_clusters = 10
loss_threshold = 0.002
bounds = []
verbosity = 0
burn = 200

# clustering choice (by Liouvllians or parameter vectors)
# vectorisation = 'Liouvillian'
vectorisation = 'parameters'
if vectorisation == 'parameters':
    #config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
    hyperparams = configs.get_hyperparams(config_name)


# container for points to cluster:
# i.e. vectorised and decomplexified Liouvillians for all models
points = []

# time trackers for profiling:
new_time = time.time()
time_last = new_time


# import separately accepted loss values and accepted proposals:
# filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R)
# with open(filename + '_accepted_proposals.pickle',
#           'rb') as filestream:
#     accepted_proposals = pickle.load(filestream)
# with open(filename + '_accepted_loss.pickle',
#           'rb') as filestream:
#     accepted_losses = pickle.load(filestream)

taken_from_each_R = []

for R in Rs:
    
    # import accepted loss values and accepted proposals from same output dictionary:
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R)
    with open(filename + '_proposals.pickle',
              'rb') as filestream:
        proposals = pickle.load(filestream)
    accepted_proposals = proposals['proposals'][burn:] # TEMP
    
    accepted_losses = [x for (x, y) in zip(proposals['loss'][1:], proposals['acceptance'])  if y == True]
    # remove losses that have no proposal saved 
    # note: (old version of code skipped saving proposals over some initial chain steps)
    accepted_losses = accepted_losses[len(accepted_losses) - len(accepted_proposals) + 1 :]
    
    print('no. proposals: ' + str(len(accepted_proposals)))
    print('no. losses: ' + str(len(accepted_losses)))
    
    
    # collect all points:
    taken_from_each_R.append((len(accepted_losses)))
    indices = list(range(len(accepted_losses)))
    #%%
    
    bounds = [
              # (40000, 41500),
              # (52000, 56000),
              # (62000, 64000),
              # (74000, 75000),
              # (83000, 85000)
              # 
              (0, len(indices))
              ]
    
    if False: # plot chain segments determined by bounds:
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
        
        
    # remove points with loss below some threshold - don't combine this with bounds!
    if False:
        print('Earlier: ' + str(len(accepted_proposals)))
        working_proposals = [x for (x, y) in zip(accepted_proposals, accepted_losses) if y < loss_threshold]
        print('After: ' + str(len(accepted_proposals)))
    
    # take only points between the specified regions (sets of bounds):
    if True:
        working_proposals = []
        for region in bounds:
            working_proposals.extend(accepted_proposals[region[0]:region[1]])
    
    if vectorisation == 'Liouvillian': # use Liouvillian
        for new_model in working_proposals:
            # build Liouvillian, turn into 1D vector, separate real and imaginary parts and concatenate:
            # note: scikit-learn cannot work with complex vectors
            Liouvillian_mat = new_model.build_Liouvillian()
            Liouvillian_vect_complex = Liouvillian_mat.ravel()
            Liouvillian_vect_separated = np.concatenate((Liouvillian_vect_complex.real, Liouvillian_vect_complex.imag))
            points.append(Liouvillian_vect_separated)
    
    elif vectorisation == 'parameters': # use model vector
        for new_model in working_proposals:
            points.append(new_model.vectorise_under_library(hyperparameters = hyperparams)[0])
        
    
    # final array to feed into clusterer 
    # note: each row a different model:
    points_array = np.stack(points)
    
    print('\n.....\ndata preparation time pre-clustering (s):' + str(np.round((new_time := time.time()) - time_last,2)) + '\n.....\n', flush = True)
    
    

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



#%% metrics to find k:

# automatically find knee/elbow aka maximum curvature point:
knee_finder = kneed.KneeLocator(clusters_counts,
                                SSEs,
                                curve="convex",
                                direction="decreasing"
                                )
elbow = knee_finder.elbow
print('\nElbow found at ' + str(elbow) + ' clusters.\n')

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
plt.title(output_name + '\nelbow found at ' + str(elbow))
plt.savefig(output_name + '_clustering_SSEs.svg')

# plot and save silhouette score vs number of clusters
plt.figure(tight_layout = True)
plt.plot(clusters_counts, silhouette_scores, 'r')
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.ylim([-0.1,1])
plt.xticks(clusters_counts, x_tick_labels)
plt.title(output_name)
plt.savefig(output_name + '_clustering_silhouette_scores.svg')

           
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
               
    
#%% use assignments and centres given with k = elbow

elbow = 4 # optional manual choice of k for assignment

final_centres = outputs_each_k[elbow]['centres']
final_assignments = outputs_each_k[elbow]['assignments']
aux = [0] + [sum(taken_from_each_R[:i+1]) for i in range(len(taken_from_each_R))]


# once again get accepted losses to plot against assignment:
for i, R in enumerate(Rs):

    # import accepted loss values and accepted proposals from same output dictionary:
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R)
    with open(filename + '_proposals.pickle',
              'rb') as filestream:
        proposals = pickle.load(filestream)
    accepted_proposals = proposals['proposals'][:] # TEMP
    accepted_losses = [x for (x, y) in zip(proposals['loss'][1:], proposals['acceptance'])  if y == True]
    # remove losses that have no proposal saved 
    # note: (old version of code skipped saving proposals over some initial chain steps)
    accepted_losses = accepted_losses[len(accepted_losses) - len(accepted_proposals) + 1 :]
    indices = list(range(len(accepted_losses)))
    
    overlay = [0 for x in range(burn)] + [x + 1 for x in final_assignments[aux[i]:aux[i+1]]]
    
    fig, ax1 = plt.subplots(tight_layout = True)
    ax1.plot(indices, accepted_losses, c = 'orange')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss', c='orange')
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.plot(indices, overlay, '--', c='blue')
    ax2.set_ylim([-0.2, elbow+0.2])
    ax2.set_ylabel('assignment', c='blue')
    ax2.set_yticks(list(range(elbow+1)))
    #plt.xticks(clusters_counts, x_tick_labels)
    #ax1.set_title('assignment to clusters')
    fig.savefig(filename + '_assignment_k' + str(elbow) + '.svg')
