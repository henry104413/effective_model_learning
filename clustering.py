#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import sklearn.cluster
import pickle
import copy
import numpy as np


# filename base
filename_base = '250420_Wit4b-grey_ForClusters_D' + str(2)

# number of runs (assuming files named starting from 1
runs = 20

# points to cluster (vectorised and decomplexified Liouvillians for all models) 
points = []

files_imported = 0

for i in range(1, runs+1):
    
    # import best model pickle file for i-th run:
    # note: occassionally files not generated... do try-except
    filename = filename_base +  '_R' + str(i) + '_best.pickle'
    try:
        with open(filename, 'rb') as filestream:
            new_model = pickle.load(filestream)
        files_imported += 1
    except FileNotFoundError:
        print(filename + '\nNOT FOUND - SKIPPING')
        continue
        
    # build Liouvillian, turn into 1D vector, separate real and imaginary parts and concatenate:
    # note: scikit-learn cannot work with complex vectors
    Liouvillian_mat = new_model.build_Liouvillian()
    Liouvillian_vect_complex = Liouvillian_mat.ravel()
    Liouvillian_vect_separated = np.concatenate((Liouvillian_vect_complex.real,
                                                 Liouvillian_vect_complex.imag))
    
    points.append(Liouvillian_vect_separated)

# ready to feed into clusterer - each row is a different model:
points_array = np.stack(points)


#%%

# define numbers of clusters explored:
min_clusters = 1
max_clusters = files_imported
clusters_counts = list(range(min_clusters, max_clusters + 1))

# containers for clustering outputs:
SSEs = []
centres = []
assignments = []
iters_required = []

# go over all cluster numbers:
for k in clusters_counts:

    # perform k-means clustering:
    kmeans = sklearn.cluster.KMeans(
        init = "random", # "random" or "k-means++"
        n_clusters = k, # number of clusters
        n_init = 50, # number of random initialisations
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


#%%

# plot SSE vs number of clusters:
import matplotlib.pyplot as plt
plt.figure()
plt.plot(clusters_counts, SSEs)
plt.xlabel('number of clusters')
plt.ylabel('SSE')
plt.xticks(clusters_counts)
plt.title(filename_base)
plt.savefig(filename_base + '_clustering_SSEs.svg')

# apparently knee at 4