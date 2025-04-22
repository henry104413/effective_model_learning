#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import sys # for passing command line arguments
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.metrics
import kneed


#%% data preparation:

# import data filename parameters and clusters number bounds from command line arguments:
# order: experiment name, defects number, run number bound, minimum clusters, maximum clusters
# note: assuming run numbers start at 1
try:
    experiment_name = str(sys.argv[1])
except:
    print('Clustering:\n Using default experiment name in filename'
          +'\n - not command line argument')
    experiment_name = ''
try:
    defects_number = str(sys.argv[2])
except:
    print('Clustering:\n Using default defects number in filename'
          +'\n - not command line argument')
    defects_number = 1
try:
    run_number_bound = int(sys.argv[3])
except:
    print('Clustering:\n Using default runs number bound in filename'
          +'\n - not command line argument')
    run_number_bound = 20
try:
    min_clusters = int(sys.argv[4])
except:
    print('Clustering:\n Using default minimum cluster number'
          +'\n - not command line argument')
    min_clusters = 2
try:
    max_clusters = int(sys.argv[5])
except:
    print('Clustering:\n Using default maximum cluster number'
          +'\n - not command line argument')
    max_clusters = False

    
# filename_base: (experiment name with defects number):
filename_base = experiment_name + '_D' + str(defects_number)


# container for points to cluster:
# i.e. vectorised and decomplexified Liouvillians for all models
points = []

# import available learned models for this defects number up to specified runs number:
files_imported = 0
for i in range(1, run_number_bound+1):
    
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

# final array to feed into clusterer 
# note: each row a different model:
points_array = np.stack(points)



#%% clustering execution:

# define numbers of clusters explored:
# note: silhouette requires at least 2 and at most points - 1
if not max_clusters: # ie. not imported from command line arguments
    max_clusters = files_imported - 1
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



#%% results:

# automatically find knee/elbow aka maximum curvature point:
knee_finder = kneed.KneeLocator(clusters_counts,
                                SSEs,
                                curve="convex",
                                direction="decreasing"
                                )
elbow = knee_finder.elbow
print('\nElbow found at ' + str(elbow) + ' clusters.\n')

# save kmeans object containing fit outputs:
with open(filename_base + '_clustering_kmeans_obj.pickle', 'wb') as filestream:
    pickle.dump(kmeans,  filestream)

# save SSE and silhouette score vs number of clusters as csv files:
np.savetxt(filename_base + '_clustering_SSEs.csv', 
           np.transpose([clusters_counts, SSEs]),
           header = 'clusters,SSEs',
           delimiter = ',', comments = '')
np.savetxt(filename_base + '_clustering_silhouette_scores.csv', 
           np.transpose(clusters_counts, silhouette_scores),
           header = 'clusters,silhouettes',
           delimiter = ',', comments = '')

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
