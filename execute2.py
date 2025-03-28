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
dataset_no = 0 # starting from 0

# extract x and y values@
contents = np.genfromtxt('Witnessing_Fig4a.csv',delimiter=',')#,dtype=float) 
dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

# sort by x:
order = xs.argsort()
xs = xs[order]
ys = ys[order]

# note: integrator errors - thought cause was unevenly spaced data likely just unsorted
# apparently works if sorted descending ascending or descending, but unsorted breaks!

# times for model evaluation:
ts = xs

# measured data feed:
measurement_datasets = [ys]
measurement_observables = ['sigmax']



#%% perform learning:
    
# if qubit initial state required:
import definitions
qubit_initial_state = definitions.ops['sigmax']

def custom_func(arg):
    print('taking abs')
    if isinstance(arg, list): return [abs(x) for x in arg]
    else: return abs(arg)


# instance of learning (quest for best model):
quest = learning_chain.LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial = (5, 2), # (qubit energy, number of defects)
                      qubit_initial_state = qubit_initial_state,
                      
                      max_chain_steps = 2000,
                      chain_step_options = {
                          'tweak all parameters': 0.1,
                          'add L': 0.05,
                          'remove L': 0.05,
                          'add qubit-defect coupling': 0.05, 
                          'remove qubit-defect coupling': 0.05,
                          'add defect-defect coupling': 0.05, 
                          'remove defect-defect coupling': 0.05
                          },
                      
                      temperature_proposal = 0.0001, # either value or (shape, scale) of gamma to sample
                      
                      jump_length_rescaling_factor = 1.05, # for scaling up or down jump lengths of parameter handler
                      
                      acceptance_window = 10,
                      acceptance_target = 0.4,
                      acceptance_band = 0.2,
                      
                      params_handler_hyperparams = { 
                          'initial_jump_lengths': {'couplings' : 0.05,
                                                   'energies' : 0.5,
                                                   'Ls' : 0.005
                                                   },
                          },
                      
                      Ls_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         'sigmax': (0.01, 0.1)
                         ,'sigmay': (0.01, 0.1)
                         ,'sigmaz': (0.01, 0.1)
                         },
                   
                      qubit2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         (('sigmax', 'sigmax'),): (0.3, 1)
                        ,(('sigmay', 'sigmay'),): (0.3, 1)
                        ,(('sigmaz', 'sigmaz'),): (0.3, 1)
                         },
                      
                      defect2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         (('sigmax', 'sigmax'),): (0.3, 1)
                        ,(('sigmay', 'sigmay'),): (0.3, 1)
                        ,(('sigmaz', 'sigmaz'),): (0.3, 1)
                        },
                      
                      params_thresholds = { # minimum values for parameters - if below then process dropped
                          # !!! does this break reversibility??                
                          'Ls':  1e-7,
                          'couplings': 1e-6
                          },
                      
                      custom_function_on_dynamics_return = False,#custom_func
                      
                      iterations_till_progress_update = 20
                      )


#%%
best = quest.run(20)

#%%
best = quest.best

costs = quest.explored_loss
acceptance_ratios = quest.chain_windows_acceptance_log
evaluation_ts = np.linspace(ts[0], ts[-1], max(10*len(ts), int(1000)))
best_datasets = best.calculate_dynamics(evaluation_ts, observable_ops = measurement_observables,
                                        custom_function_on_return = False)


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
# maybe add additional check to in unlikely case parallel run ends up with same name...

# create outputs:
output.Output(toggles = Toggles, filename = timestamp,
       dynamics_ts = [ts, evaluation_ts],
       dynamics_datasets = [measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['measured', 'learned'],
       dynamics_formatting = ['b+', 'r-'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       acceptance = acceptance_ratios,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


#%% 




