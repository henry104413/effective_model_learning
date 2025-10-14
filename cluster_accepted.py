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

# settings and source data:
experiment_name = '250818-sim-1T-4JL-2tweak' + '_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 2
Rs = [2] # [1 + i for i in range(5)]
hyperparams = configs.get_hyperparams(config_name)
output_name = experiment_name + '_clust_test3_thresh0_001'
min_clusters = 2
max_clusters = 6
loss_threshold = 0.001

# container for points to cluster:
# i.e. vectorised and decomplexified Liouvillians for all models
points = []

# collect all points:
for R in Rs:
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R)
    with open(filename + '_accepted_proposals.pickle',
              'rb') as filestream:
        accepted_proposals = pickle.load(filestream)
    
    if loss_threshold > 0:
        with open(filename + '_accepted_loss.pickle',
                  'rb') as filestream:
            accepted_losses = pickle.load(filestream)
        print('Earlier: ' + str(len(accepted_proposals)))
        accepted_proposals = [x for (x, y) in zip(accepted_proposals, accepted_losses) if y < loss_threshold]
        print('After: ' + str(len(accepted_proposals)))
    
    for new_model in accepted_proposals:
        # build Liouvillian, turn into 1D vector, separate real and imaginary parts and concatenate:
        # note: scikit-learn cannot work with complex vectors
        Liouvillian_mat = new_model.build_Liouvillian()
        Liouvillian_vect_complex = Liouvillian_mat.ravel()
        Liouvillian_vect_separated = np.concatenate((Liouvillian_vect_complex.real, Liouvillian_vect_complex.imag))
        points.append(Liouvillian_vect_separated)
    
# final array to feed into clusterer 
# note: each row a different model:
points_array = np.stack(points)

# raise SystemExit()
#%% clustering execution:

clusters_counts = list(range(min_clusters, max_clusters + 1))

# containers for clustering outputs:
SSEs = []
centres = []
assignments = []
iters_required = []
silhouette_scores = [] # higher is better

# go over all cluster numbers:
for k in clusters_counts:

    # perform k-means clustering:
    kmeans = sklearn.cluster.KMeans(
        init = "random", # "random" or "k-means++"
        n_clusters = k, # number of clusters
        n_init = 10, # number of random initialisations
        max_iter = 300, # maximum centroid adjustments 
        # note: usually converges in just a handful of iterations
        random_state = None # just leave this
        )
    kmeans.fit(points_array)
    
    # save outputs for this k:
    SSEs.append(kmeans.inertia_)
    centres.append(kmeans.cluster_centers_)
    assignments.append(kmeans.labels_)
    iters_required.append(kmeans.n_iter_)
    silhouette_scores.append(sklearn.metrics.silhouette_score(points_array, kmeans.labels_))



#%% results:

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
    x_tick_labels[elbow - clusters_counts[0]] = elbow

# plot SSE vs number of clusters:
plt.figure(tight_layout = True)
plt.plot(clusters_counts, SSEs, 'b')
plt.xlabel('number of clusters')
plt.ylabel('SSE')
plt.xticks(ticks = clusters_counts, labels = x_tick_labels)
plt.title(output_name + '\nelbow found at ' + str(elbow))
plt.savefig(output_name + '_clustering_SSEs.svg')

# plot silhouette score vs number of clusters
plt.figure(tight_layout = True)
plt.plot(clusters_counts, silhouette_scores, 'r')
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.xticks(clusters_counts, x_tick_labels)
plt.title(output_name)
plt.savefig(output_name + '_clustering_silhouette_scores.svg')
