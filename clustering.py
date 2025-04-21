#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.metrics
import kneed

# filename = '2025_04_09_132733_1_3_best.pickle'
filename = '250420_Wit4b-grey_ForClusters_D1_R1_best.pickle'
with open(filename, 'rb') as filestream:
    new_model = pickle.load(filestream)
raise SystemExit()


# filename base
filename_base = '250420_Wit4b-grey_ForClusters_D' + str(2)


# number of runs (assuming files named starting from 1)
runs = 20

# points to cluster (vectorised and decomplexified Liouvillians for all models) 
points = []

# import available learned models from all runs:
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
    except Exception as e:
        print('Error: ' + repr(e))
        continue
        
    # build Liouvillian, turn into 1D vector, separate real and imaginary parts and concatenate:
    # note: scikit-learn cannot work with complex vectors
    Liouvillian_mat = new_model.build_Liouvillian()
    Liouvillian_vect_complex = Liouvillian_mat.ravel()
    Liouvillian_vect_separated = np.concatenate((Liouvillian_vect_complex.real,
                                                 Liouvillian_vect_complex.imag))
    
    points.append(Liouvillian_vect_separated)

# array to feed into clusterer - each row a different model:
points_array = np.stack(points)


#%%

# define numbers of clusters explored:
min_clusters = 2 # note: silhouette requires at least 2
max_clusters = files_imported - 1 # note: silhouette requires at most points - 1
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
    silhouette_scores.append(sklearn.metrics.silhouette_score(points_array, kmeans.labels_))
    
#from sklearn.metrics import silhouette_score
#from kneed import KneeLocator
#...after running something like: conda install -c conda-forge kneed


#%%

# find knee/elbow aka maximum curvature point:
knee_finder = kneed.KneeLocator(clusters_counts,
                                SSEs,
                                curve="convex",
                                direction="decreasing"
                                )
elbow = knee_finder.elbow
print('\nElbow found at ' + str(elbow) + ' clusters.\n')

# plot SSE vs number of clusters:
plt.figure()
plt.plot(clusters_counts, SSEs, 'b')
plt.xlabel('number of clusters')
plt.ylabel('SSE')
plt.xticks(clusters_counts)
plt.title(filename_base + '\nelbow found at ' + str(elbow))
plt.savefig(filename_base + '_clustering_SSEs.svg')

# plot silhouette score vs number of clusters
plt.figure()
plt.plot(clusters_counts, silhouette_scores, 'r')
plt.xlabel('number of clusters')
plt.ylabel('silhouette score')
plt.xticks(clusters_counts)
plt.title(filename_base)
plt.savefig(filename_base + '_clustering_silhouette_scores.svg')