#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""
from __future__ import annotations
#import os
#import pickle
#import copy
import typing
import matplotlib.pyplot as plt
import numpy as np
if typing.TYPE_CHECKING:
    #from basic_model import BasicModel
    #from learning_model import LearningModel
    pass


experiment_name = '250602-full'
dataset_name = "Wit-Fig4-6-0_025"
defects_numbers = [0, 1, 2, 3] # list of all
knees = [6, 7, 6, 12] # manually input identified knees for corresponding defects

def load_dataset(filenamename: str) -> tuple[np.ndarray]:
    """
    Loads data from dataset_name.csv in current folder,
    returns tuple of arrays: xs, ys.
    
    Takes first pair of columns, filters out non-numbers, sorts by x,
    !! also rescales xs by 1/1000 (from ns to us in target data).
    """
    
    # choose dataset:
    datafile = filenamename + '.csv'

    # extract x and y values
    contents = np.genfromtxt(datafile,delimiter=',')#,dtype=float) 
    dataset = contents[:,[0, 1]]
    xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
    ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

    # sort by x:
    order = xs.argsort()
    xs = xs[order]
    ys = ys[order]
    
    return xs, ys



#%% plots:
    
plt.rcParams["font.size"] = 16


    
# training on full dataset - best candidate on top of target dataset, given D: 

labels = {'SSEs': 'SSE',
          'silhouette_scores': 'silhouette score'}

for metric in ['SSEs', 'silhouette_scores']:

    colours = (x for x in ['b', 'r', 'g', 'm', 'k'])
        
    plt.figure()
    plt.ylabel(labels[metric])
    plt.xlabel(r'$k$')
    #plt.ylim([-0.05, 1.05])
    plt.xlim([1, 21])
    
    if metric == 'SSEs': plt.yscale('log')


    for i, defects_number in enumerate(defects_numbers):
        
        
        filename_base = (experiment_name + '-' + dataset_name
                         + '_D' + str(defects_number)+ '_clustering_')
    
        # load target data corresponding to dataset name in candidate_models_set instance
        # note: assumed present in current folder
        ks, vals = load_dataset(filename_base + metric)
        
        
        
        colour = next(colours)
        plt.plot(ks, vals, '-' + colour
                ,label = str(defects_number)
                ,alpha = 0.4, linewidth = 2
                )
        plt.plot(knees[i], vals[np.where(ks == knees[i])], 'o' + colour, 
                 alpha = 0.4, markeredgewidth = 2, markersize = 10)
        
    legend = plt.legend(title = r'$D' + '$')
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    if False: plt.text()
    plt.savefig(experiment_name + '-' + dataset_name + '_' + metric
                + '.svg', dpi = 1000, bbox_inches='tight')

    
