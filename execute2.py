#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import multiprocessing
import time
import numpy as np

import basic_model
import learning_chain
import output



#%% import target data:
    
# import data from CSV file with possible annotations skipped
# assuming subsequent pairs of columns are different datasets 
# and numberical entries on single row are x, y values

# choose data:
datafile = 'Witnessing_Fig4a.csv'
dataset_no = 2 # starting from 0

# extract x and y values@
contents = np.genfromtxt('Witnessing_Fig4a.csv',delimiter=',')#,dtype=float) 
dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

# sort by x:
order = xs.argsort()
xs = xs[order]
ys = ys[order]

ts = np.linspace(min(xs), max(xs), int(100))

ys_interp = np.interp(ts, xs, ys)

ys_mod = ys_interp*2 - 1

measurement_datasets = [ys_mod]
measurement_observables = ['sigmax']


import matplotlib.pyplot as plt
plt.figure()
#plt.plot(xs, ys, 'bo')
plt.plot(ts, ys_mod, 'r+')



#%% perform learning:
    
# if qubit initial state required:
import definitions
qubit_initial_state = definitions.ops['sigmax']


# instance of learning (quest for best model):
quest = learning_chain.LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial = (5, 1), # (qubit energy, number of defects)
                      qubit_initial_state = qubit_initial_state,
                      
                      max_chain_steps = 1000,
                      chain_step_options = {
                          'tweak all parameters': 0.2,
                          'add L': 0.05,
                          'remove L': 0.05,
                          'add qubit-defect coupling': 0.05, 
                          'remove qubit-defect coupling': 0.05,
                          'add defect-defect coupling': 0.025, 
                          'remove defect-defect coupling': 0.025
                          },
                      
                      temperature_proposal_shape = 0.01, # aka k
                      temperature_proposal_scale = 0.01, # aka theta
                      
                      jump_length_rescaling_factor = 1.05, # for scaling up or down jump lengths of parameter handler
                      
                      acceptance_window = 10,
                      acceptance_target = 0.4,
                      acceptance_band = 0.2,
                      
                      params_handler_hyperparams = { 
                          'initial_jump_lengths': {'couplings' : 0.01,
                                                   'energy' : 0.1,
                                                   'Ls' : 0.001
                                                   },
                          },
                      
                      Ls_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         'sigmax': (0.01, 0.1)
                         ,'sigmay': (0.01, 0.1)
                         ,'sigmaz': (0.01, 0.1)
                         },
                   
                      qubit2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         (('sigmax', 'sigmax'),): (0.2, 0.5)
                        ,(('sigmay', 'sigmay'),): (0.2, 0.5)
                        ,(('sigmaz', 'sigmaz'),): (0.2, 0.5)
                         },
                      
                      defect2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         (('sigmax', 'sigmax'),): (0.2, 0.5)
                        ,(('sigmay', 'sigmay'),): (0.2, 0.5)
                        ,(('sigmaz', 'sigmaz'),): (0.2, 0.5)
                        }
                      )


#%%
best = quest.run(1000)

#%%
best = quest.best

costs = quest.explored_loss
acceptance_ratios = quest.chain_windows_acceptance_log
best_datasets = best.calculate_dynamics(ts, observable_ops = measurement_observables)


#%% chain run outputs:

# output controls bundle:
class Toggles():    
    comparison = True # plot comparison of dynamics
    loss = True # plot cost function progression
    acceptance = True # plot acceptance ratios over subsequenct windows
    graphs = False # plot model graphs with corresponding labels
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True # save chain hyperparameters as json

# unique name (date and time stamp):
timestamp = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())

# create outputs:
output.Output(toggles = Toggles, filename = timestamp,
       dynamics_ts = ts,
       dynamics_datasets = [measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['measured', 'learned'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       acceptance = acceptance_ratios,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


#%% 




