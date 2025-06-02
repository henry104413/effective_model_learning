#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""
from __future__ import annotations
import os
import pickle
import copy
import typing
if typing.TYPE_CHECKING:
    #from basic_model import BasicModel
    from learning_model import LearningModel


experiment_name = 'combined-'
defects_number = 1

# candidate sets:
# list of tuples (experiment_name, defects_number)
candidates_sets_descriptors = [('combined-', 1)
                               #
                              ]

# first make list of all files in folder with this name base? (so across all found Rs)
# then load the best models
# pick the one R with lowest cost
# take corresponding data

# probably class where each instance is one such bunch? (ie for each instance one best model)

class CandidatesSet:
    
# instance consists of experiment name and D
# R candidate models have been learned (repetitions)
# method pull_all to go through current folder and loads all Rs found for this experiment name and D
# it then identifies, saves, and returns the best candidate model (pickle with actual model)

# method find best will look at all models in folder, load each, but only save the current best one
# at the end only one should be loaded as the champion of the candidate set

# it will also hold the target data and times
# and generate its own time array and evaluate the model across it

# thus each bunch (change name maybe) can immeadiately plot a curve comparison of best model for target dataset, given D
    

    def __init__(self, experiment_name: str,
                 defects_number: int):
        
        # experiment and defects number defining candidate set in current folder:
        self.experiment_name = experiment_name
        self.defects_number = defects_number
        
    def pull_all(self):
    # to load all matching models - currently unused - could be used in clusterer maybe
        pass
    
    def find_best(self) -> LearningModel:
        """
        Finds, returns and saves best candidate model from this candidate set, 
        which is all files from current folder 
        with matching experiment name and defects number.
        """
        
        # get candidate models file names:
        candidates_files = self.get_model_paths()
        
        # containers:
        self.best_loss = None
        self.best_candidate = None
        self.best_filename = None
        
        # go over all candidate files, import, save if improvement:
        for candidate_file in candidates_files:
            with open(candidate_file, 'rb') as filestream:
                candidate = pickle.load(filestream)
            if not self.best_loss: 
                # ie. no candidate saved yet - update to first
                self.best_loss = candidate.final_loss
                self.best_candidate = copy.deepcopy(candidate)
                self.best_filename = candidate_file
            else: 
                # ie. candidate exists, compare
                if (temp := candidate.final_loss) < self.best_loss: 
                    # ie. improvement - update best candidate
                    self.best_loss = temp
                    self.best_candidate = copy.deepcopy(candidate)
                    self.best_filename = candidate_file
                
        return self.best_candidate        
    
    def get_model_paths(self) -> list[str]:
        """
        Saves and returns list of pickled models filenames from current folder
        that match experiment name and defects number of this candidate set.
        """
        
        all_files = os.listdir()
        self.matching_model_files = [x for x in all_files 
                          if ((self.experiment_name + '_D' + str(self.defects_number)) in x)
                          and ('_best.pickle' in x)]
        return self.matching_model_files



# for each candidacte set specified by experiment name and defects number 
# (ie. all repetitions of same model learning), instantiate CandidateSet and run find_best()
# so that a champion candidate for each learning configuration is available:
candidates_sets = [] # list of candidate set instances, each with its champion:
for candidates_set_descriptor in candidates_sets_descriptors:
    candidates_set = CandidatesSet(candidates_set_descriptor[0], candidates_set_descriptor[1])
    candidates_set.find_best()
    candidates_sets.append(candidates_set)






#%% plot what needs plotted together:
 
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(dataset_name: str) -> tuple[np.ndarray]:
    """
    Loads data from dataset_name.csv in current folder,
    returns tuple of arrays: xs, ys.
    
    Takes first pair of columns, filters out non-numbers,
    sorts by x, rescales xs by 1/1000 (from ns to us for target data).
    """
    
    # choose data:
    datafile = dataset_name + '.csv'
    dataset_no = 0 
    # note: 0 means first pair of columns
    # note: currently assuming first dataset in file is target

    # extract x and y values
    contents = np.genfromtxt(datafile,delimiter=',')#,dtype=float) 
    dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
    xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
    ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

    # rescale xs (ie. time, from ns to us)
    xs = xs/1000

    # sort by x:
    order = xs.argsort()
    xs = xs[order]
    ys = ys[order]
    
    return xs, ys


dataset_name = 'Wit-Fig4-6-0_025'

xs, ys = load_dataset(dataset_name)

plt.figure()
plt.ylabel(r'<$\sigma_x$>')
plt.xlabel(r'time ($\mu s$)')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.005, 2.505])
    
# plot all the datasets in the comparison for this observable:
# assumed times may differ for datasets but same across each dataset for all observables
plt.plot(xs, ys, '+b', 
        label = dataset_name
        ,markersize = 10, markeredgewidth = 2, linewidth = 4
        )            
plt.legend()
if False:
    plt.text()
plt.savefig(dataset_name + '_testplot.svg', dpi = 1000, bbox_inches='tight')


