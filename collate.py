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
import matplotlib.pyplot as plt
import numpy as np
if typing.TYPE_CHECKING:
    #from basic_model import BasicModel
    from learning_model import LearningModel


# for multiple datasets and defects numbers under one experiment,
# specify varying parameter (restoring or damping - both frequencies),
# then make list of descriptor tuples (experiment_name, dataset_name, defects_number):
# note: for each descriptor, multiple repetitions (identical parameter chains) were run

varying = 'damping'
experiment_name = '250602-full'
dataset_names = {'restoring': ["Wit-Fig4-5-0_1", "Wit-Fig4-6-0_1", "Wit-Fig4-7-0_1"],
                 'damping': ["Wit-Fig4-6-0_025", "Wit-Fig4-6-0_1", "Wit-Fig4-6-0_2"]
                 }
defects_numbers = [2] #[0, 1, 2] # list of all

descriptors = [(experiment_name, T, D) for T in dataset_names[varying] for D in defects_numbers]
labels = {}
labels['damping'] = {
          "Wit-Fig4-5-0_1" : '0.1 MHz',
          "Wit-Fig4-6-0_025" : '0.025 MHz',
          "Wit-Fig4-6-0_1" : '0.1 MHz',
          "Wit-Fig4-6-0_2" : '0.2 MHz',
          "Wit-Fig4-7-0_1" : '0.1 MHz'
         }
# frequency as in inverse of the decay time
labels['restoring'] = {
          "Wit-Fig4-5-0_1" : '5 MHz',
          "Wit-Fig4-6-0_025" : '6 MHz',
          "Wit-Fig4-6-0_1" : '6 MHz',
          "Wit-Fig4-6-0_2" : '6 MHz',
          "Wit-Fig4-7-0_1" : '7 MHz'
         }


#%%


# first make list of all files in folder with this name base? (so across all found Rs)
# then load the best models
# pick the one R with lowest cost
# take corresponding data

# probably class where each instance is one such bunch? (ie for each instance one best model)

class CandidateModelsSet:
    
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
                 dataset_name: str,
                 defects_number: int):
        
        # experiment name, target dataset name, and defects number
        # (define candidate set in current folder):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
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
        that match experiment_name-dataset_name and defects number of this candidate set.
        
        !! Note: The naming convention for the experiment and dataset name here has a HYPHEN.
        """
        
        all_files = os.listdir()
        self.matching_model_files = [x for x in all_files 
                          if ((self.experiment_name + '-' + self.dataset_name
                               + '_D' + str(self.defects_number)) in x)
                          and ('_best.pickle' in x)]
        return self.matching_model_files



def load_dataset(dataset_name: str) -> tuple[np.ndarray]:
    """
    Loads data from dataset_name.csv in current folder,
    returns tuple of arrays: xs, ys.
    
    Takes first pair of columns, filters out non-numbers, sorts by x,
    !! also rescales xs by 1/1000 (from ns to us in target data).
    """
    
    # choose dataset:
    datafile = dataset_name + '.csv'

    # extract x and y values
    contents = np.genfromtxt(datafile,delimiter=',')#,dtype=float) 
    dataset = contents[:,[0, 1]]
    xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
    ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

    # rescale xs (ie. time, from ns to us) and sort by x:
    xs = xs/1000
    order = xs.argsort()
    xs = xs[order]
    ys = ys[order]
    
    return xs, ys



# for each candidate models set specified by experiment name, target dataset name.
# and defects number (ie. all repetitions of same model learning),
# instantiate CandidateModelsSet and run its find_best(),
# so that a champion candidate for each learning configuration is available:
candidate_models_sets = [] # list of instances for all descriptors, each has champion:
for descriptor in descriptors:
    candidate_models_set = CandidateModelsSet(descriptor[0], descriptor[1], descriptor[2])
    candidate_models_set.find_best()
    candidate_models_sets.append(candidate_models_set)






#%% plots:
    
plt.rcParams["font.size"] = 16


    
# training on full dataset - best candidate on top of target dataset, given D: 

plt.figure()
plt.ylabel(r'<$\sigma_x$>')
plt.xlabel(r'time ($\mu s$)')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 2.55])
 
colours = (x for x in ['b', 'r', 'g', 'm', 'k'])
for candidate_set in candidate_models_sets:
    
    # load target data corresponding to dataset name in candidate_models_set instance
    # note: assumed present in current folder
    xs, ys = load_dataset(candidate_set.dataset_name)
    
    champion = candidate_set.best_candidate
    
    
    colour = next(colours)
    plt.plot(xs, ys, '.' + colour
            #,label = 'measurements'
            ,alpha = 0.5, markersize = 6, markeredgewidth = 1
            )
    plt.plot(xs, champion.calculate_dynamics(xs, ['sigmax'])[0], '-' + colour
            ,label = labels[varying][candidate_set.dataset_name]
            ,alpha = 0.4, linewidth = 2
            )
    
    legend = plt.legend(title = r'$D=' + str(candidate_set.defects_number) + '$' + ', ' + varying)
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    if False: plt.text()
    plt.savefig('varying_' + varying 
                + '_D' + str(candidate_set.defects_number)
                +'.svg', dpi = 1000, bbox_inches='tight')


