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

varying = 'all'
experiment_name = '250602-full'
dataset_names = {'restoring': ["Wit-Fig4-5-0_1", "Wit-Fig4-6-0_1", "Wit-Fig4-7-0_1"],
                 'damping': ["Wit-Fig4-6-0_025", "Wit-Fig4-6-0_1", "Wit-Fig4-6-0_2"],
                 'all' : ["Wit-Fig4-5-0_1", "Wit-Fig4-6-0_025",
                          "Wit-Fig4-6-0_1", "Wit-Fig4-6-0_2",
                          "Wit-Fig4-7-0_1"]
                 }
defects_numbers = [ 1, 2, 3] # list of all

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

feature_sizes_dict = { 
    # from bottom to top of first oscillation (first two local extrema difference after initial descent)
          "Wit-Fig4-5-0_1" : 0.286,
          "Wit-Fig4-6-0_025" : 0.468,
          "Wit-Fig4-6-0_1" : 0.181,
          "Wit-Fig4-6-0_2" : 0.046,
          "Wit-Fig4-7-0_1" : 0.102
         }




#%%


# first make list of all files in folder with this name base? (so across all found Rs)
# then load the best models
# pick the one R with lowest cost
# take corresponding data

# probably class where each instance is one such bunch? (ie for each instance one best model)

class CandidateModelsSet:
    
# instance consists of experiment base name and D
# R candidate models have been learned (repetitions)
# method pull_all to go through current folder and loads all Rs found for this experiment name and D
# it then identifies, saves, and returns the best candidate model (pickle with actual model)

# method find best will look at all models in folder, load each, but only save the current best one
# at the end only one should be loaded as the champion of the candidate set

# it will also hold the target data and times
# and generate its own time array and evaluate the model across it

# thus each bunch (change name maybe) can immeadiately plot a curve comparison of best model for target dataset, given D
    

    def __init__(self, experiment_name: str,
                 #dataset_name: str,
                 defects_number: int):
        
        # experiment name, target dataset name, and defects number
        # (define candidate set in current folder):
        self.experiment_name = experiment_name
        #self.dataset_name = dataset_name
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
        
        if True: # print
            print('\n\nchampion for ' + str(self.experiment_name)
                  #+ '-' + str(self.dataset_name)
                  + '_D' + str(self.defects_number)
                  + ':\n file = ' + str(self.best_filename)
                  + ':\n loss = ' + str(self.best_loss) )
    
        return self.best_candidate        
    
    def get_model_paths(self) -> list[str]:
        """
        Saves and returns list of pickled models filenames from current folder
        that match experiment_name-dataset_name and defects number of this candidate set.
        
        !! Note: The naming convention here earlier was experiment_name HYPHEN dataset_name.
        Now changed to just experiment name and feeding it the full name base up to _D.
        """
        
        all_files = os.listdir()
        self.matching_model_files = [x for x in all_files 
                          if ((self.experiment_name 
                               #+ '-' + self.dataset_name
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



def return_champion_loss(experiment_name, D):
    """
    Returns loss (float number) of champion of set of repetitions;
    arguments: experiment name, defects number (cycles through repetitions available in folder). 
    
    Note: Earlier was experiment and dataset names separately
    but merged into one base name to be passed up to _D.
    """
    candidate_models_set = CandidateModelsSet(experiment_name, D)
    candidate_models_set.find_best()
    champion_loss = candidate_models_set.best_loss
    return champion_loss



#%% training on full dataset - best candidate on top of target dataset, given D: 

# note: initially plotted individually for each D... I think?    

# for each candidate models set specified by experiment name, target dataset name.
# and defects number (ie. all repetitions of same model learning),
# instantiate CandidateModelsSet and run its find_best(),
# so that a champion candidate for each learning configuration is available:
candidate_models_sets = [] # list of instances for all descriptors, each has champion:
for descriptor in []:# descriptors:
    candidate_models_set = CandidateModelsSet(descriptor[0], descriptor[1], descriptor[2])
    candidate_models_set.find_best()
    candidate_models_sets.append(candidate_models_set)

# plots:
    
plt.rcParams["font.size"] = 16    

if not True:    
    
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
    


#%% checkerboard of champions' final loss across different Ds and different datasets sorted by feature size:
   
plt.rcParams["font.size"] = 16

    
# dataset names sorted by feature size
vals = np.array([val for key, val in feature_sizes_dict.items()])
order = vals.argsort()
feature_sizes = vals[order]
datasets_by_feature_size = np.array([x for x in feature_sizes_dict])[order]

Ds = np.array(defects_numbers)


if not True: # make matrix A

    # make array of champions final loss functions (matrix to plot essentially):
    # ...
    A = np.empty([len(datasets_by_feature_size), len(Ds)])    
    for row, dataset in enumerate(datasets_by_feature_size):
        for col, D in enumerate(Ds):
            A[row, col] = return_champion_loss(experiment_name, dataset, D)
            

if not True: # make unwrapped list:

    comps = [(T, D) for T in datasets_by_feature_size for D in Ds]  
    xs = [x[1] for x in comps]
    ys = [feature_sizes_dict[x[0]] for x in comps]
    zs = [return_champion_loss(experiment_name, x[0], x[1]) for x in comps]
    

#%% 
if not True: # plot matrix
    
    fig = plt.figure()
    ax = plt.gca()
    cax = ax.matshow(A, interpolation='nearest')
    fig.colorbar(cax)
    
    feature_size_labels = [str(x) for x in feature_sizes]
    D_labels = [str(x) for x in Ds]
    
    yaxis = np.arange(len(feature_size_labels))
    xaxis = Ds
    ax.set_yticks(yaxis)
    ax.set_xticks(Ds)
    ax.set_yticklabels(feature_size_labels)
    ax.set_xticklabels(D_labels)   
    
    ax.set_xlabel(r'$D$')
    ax.set_ylabel('feature size')     
    ax.set_title('final loss')        
    fig.savefig('test_checkerboard' + '.svg')
  
#%% 
if not True: # plot as scatter thing:
    
    # normalise z to order of 10:
    zs = [z/max(zs) for z in zs]
    
    import matplotlib.ticker
    fmt = lambda x, pos: '{:.1f}'.format(x)
    

    
    fig = plt.figure(figsize=(2.5, 4.2))
    import matplotlib.colors
    plt.scatter(xs, ys, c = zs,
                edgecolors='none',
                s=0.5e3, marker='s',
                alpha = 1,
                cmap = 'inferno',
                #norm = matplotlib.colors.LogNorm()
                )
    #cb.ax.set_yticklabels([("%d" % i) for i in cb.get_ticks()]) # set ticks of your format
    ax = plt.gca()
    ax.set_facecolor('#D3D3D3')
    #ax.tick_params(axis='x', pad=)
    plt.xlabel(r'$D$')
    plt.ylabel('feature size')     
    plt.ylim([-0.00, 0.55])
    plt.xlim([0.2, 3.8])
    formatter = matplotlib.ticker.LogFormatter() # add to colorbar: format = formatter
    #formatter = matplotlib.ticker.ScalarFormatter()
    cbar = plt.colorbar(label = 'relative loss', format = '{x:.1f}')
    # formatting not working for LogNorm - only formats first tick... formatter doesn't do anything
    #plt.xticks(xs) # works but for some reason leads to bold font... jeez matplotlib honestly
    
    #cb.update_ticks()
    fig.savefig('checkerboard' + '.svg', dpi = 1000, bbox_inches='tight')
  
    
#%%
if True: # scatter of custom sets with different experiment names:
    
    plt.rcParams["font.size"] = 16

    # old configs for 250615 batch

    # configs_OG = [
    #     '250615-L-qubit-only-everything-couples-Wit-Fig4-6-0_025_D2',
    #     '250615-L-qubit-only-single-full-library-Wit-Fig4-6-0_025_D1',
    #     '250615-L-qubit-only-starlike-Wit-Fig4-6-0_025_D2',
    #     '250615-L-qubit-sigmaz-only-everything-couples-Wit-Fig4-6-0_025_D2',
    #     '250615-L-qubit-sigmaz-only-single-sigmax-coupling-Wit-Fig4-6-0_025_D1',
    #     '250615-L-qubit-sigmaz-only-starlike-Wit-Fig4-6-0_025_D2',
    #     '250602-full-Wit-Fig4-6-0_025_D1',
    #     '250602-full-Wit-Fig4-6-0_025_D2'
    #         ]
    
    # configs = { # name of experiment : defect numbers (tuple - must be iterable anyway!)
    #     '250615-L-qubit-only-everything-couples' : (2,),
    #     '250615-L-qubit-only-single-full-library' : (1,),
    #     '250615-L-qubit-only-starlike' : (2,),
    #     '250615-L-qubit-sigmaz-only-everything-couples' : (2,),
    #     '250615-L-qubit-sigmaz-only-single-sigmax-coupling' : (1,),
    #     '250615-L-qubit-sigmaz-only-starlike' : (2,),
    #     '250602-full' : (1, 2)
    #           }
   #  sep = '\n'
   # # $''\n'r'$
   #  experiment_labels = { # presentable labels corresponding to experiments ie. configs keys
      
   #      '250615-L-qubit-only-everything-couples' : 
   #          r'$L_{syst} \in \{\sigma_x, \sigma_y, \sigma_z\}$''\n'r'$L_{virt} \in \{\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x, \sigma_y \sigma_y, \sigma_z \sigma_z\}$'+sep+'mesh',
   #      '250615-L-qubit-only-single-full-library' :
   #          r'$L_{syst} \in \{\sigma_x, \sigma_y, \sigma_z\}$''\n'r'$ L_{virt} \in \{\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x, \sigma_y \sigma_y, \sigma_z \sigma_z\}$'+sep+'star',
   #      '250615-L-qubit-only-starlike' :
   #          r'$L_{syst} \in \{\sigma_x, \sigma_y, \sigma_z\}$''\n'r'$L_{virt} \in \{\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x, \sigma_y \sigma_y, \sigma_z \sigma_z\}$'+sep+'star',
   #      '250615-L-qubit-sigmaz-only-everything-couples' :
   #          r'$L_{syst} \in \{\sigma_z\}$''\n'r'$L_{virt} \in \{\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x, \sigma_y \sigma_y, \sigma_z \sigma_z\}$'+sep+'mesh',
   #      '250615-L-qubit-sigmaz-only-single-sigmax-coupling' : 
   #          r'$L_{syst} \in \{\sigma_z\}$''\n'r'$L_{virt} \in \{\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x\}$'+sep+'star',
   #      '250615-L-qubit-sigmaz-only-starlike' : 
   #          r'$L_{syst} \in \{\sigma_z\}$''\n'r'$L_{virt} \in \{\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x, \sigma_y \sigma_y, \sigma_z \sigma_z\}$'+sep+'star',
   #      '250602-full' :
   #          r'$L_{syst} \in \{\sigma_x, \sigma_y, \sigma_z\}$''\n'r'$L_{virt} \in \{\sigma_x, \sigma_y, \sigma_z\}$'+sep
   #         +r'$C \in \{\sigma_x \sigma_x, \sigma_y \sigma_y, \sigma_z \sigma_z\}$'+sep+'mesh'
   #            }
    
    losses = []
    labels = []
    
    configs = ['Lsyst-sz-Lvirt--Cs2v-sx-Cv2v--', 
               'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v-sx-', 
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v--', 
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v-sx-',
               'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx-Cv2v--',
               'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx-Cv2v-sx-',
               'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-',
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--',
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-',
               'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx,sy,sz-Cv2v--', 
               'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-',
               'Lsyst-sx-Lvirt--Cs2v-sx-Cv2v--',
               'Lsyst-sx-Lvirt--Cs2v-sz-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sz-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sx,sy-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sx,sz-Cv2v--'
               ]
    
    colours = ['orange' if x < 12 else 'firebrick' if x < 15 else 'navy' for x in range(17)]
    # configs = [
    #     'Lsyst-sx-Lvirt--Cs2v-sx-Cv2v--',
    #     'Lsyst-sx-Lvirt--Cs2v-sz-Cv2v--',
    #     'Lsyst-sz-Lvirt--Cs2v-sz-Cv2v--',
    #     'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v--',
    #     'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--'
    #     ]
    
    experiment_base = '250624'
    target_file = 'Wit-Fig4-6-0_025'
    for config in configs: 
        experiment_name = experiment_base + '_' + target_file + '_' + config 
    
        for D in [2]: # tuple of Ds   
        
            losses.append(return_champion_loss(experiment_name, D))
            labels.append(config)
            
    def ops_label_dresser(ops: str, setname: str) -> str: # works
        """
        Argument is comma-delimiting string containing sx, sy, sz.
        
        Returns dressed string of operators to show in as processes of each class,
        assuming setname containing "C" means it's symmetric coupling (ie. applied on both subsystems).
        """
        ops_list = [x for x in ops.split(',') if x] # separates by comma and eliminates empty strings
        symbols = {'sx': r'$\sigma_x$',
                   'sy': r'$\sigma_y$',
                   'sz': r'$\sigma_z$'}
        output = ''
        for x in ops_list:
            if output: output = output + ', '
            output = output + symbols[x]
            if 'C' in setname: output = output + r'$\otimes$' + symbols[x] 
        return output
        
    def interpret_config(string: str) -> str:
        """
        Argument is configuration label string of form Lsyst-ops-Lvirt-ops-Cs2v-ops-Cv2v-ops-,
        where ops is comma-delimited string containg any of sx, sy, sz.
        
        Returns dressed label of Ls and couplings as sets of operators;
        each single operator taken as acting on both systems if coupling.
        """
        Lsyst_Lvirt_Cs2v_Cv2v = [string.split('-')[x] for x in (1,3,5,7)]
        setnames = [r'$L_{syst}$', r'$L_{virt}$', r'$C_{s2v}$', r'$C_{v2v}$']
        label = ''
        for i, ops in enumerate(Lsyst_Lvirt_Cs2v_Cv2v):
            if i>0: label = label + '\n'
            label = label + setnames[i] + r'$\in\{$' + ops_label_dresser(ops, setnames[i]) + r'$\}$'
        
        return label    
            
    A = ([interpret_config(x) for x in labels])
            
        
        
        
  #%%  
    #losses = [x/max(losses) for x in losses]
    losses_arr = np.array(losses)
    order = losses_arr.argsort()
    labels_arr = np.array([interpret_config(x) for x in labels])
    losses_arr_sorted = losses_arr[order]
    labels_arr_sorted = labels_arr[order]
    colours = np.array(colours)[order]
    
    labels_sorted = [labels[i] for i in order]
    
    plt.figure(figsize=(10, 30))
    plt.barh(np.arange(len(losses)), losses_arr_sorted, color=colours)
    #plt.xscale('log')
    plt.yticks(ticks = np.arange(len(losses)), labels = labels_arr_sorted)
    plt.title('D = 2')
    plt.xlabel('lowest loss')#
    ax=plt.gca()
    #ax.axis["left"].major_ticklabels.set_ha("left")
    #ax.set_xticklabels(ha='left')
    plt.savefig('250624_3_loss_vs_configuration_nonlog' + '.svg', dpi = 1000, bbox_inches='tight')
  
    