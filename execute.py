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



#%% generate simulated target data:

import definitions
qubit_initial_state = definitions.ops['sigmax']
    
# set up ground truth model:  
GT = basic_model.BasicModel()
GT.add_TLS(TLS_label = 'qubit',
           is_qubit = True,
           initial_state = qubit_initial_state,
           energy = 5,
           couplings = {
               
                        },
           Ls = {
                 'sigmaz' : 0.0001,
                 'sigmax' : 0.003
                 
                 }
           )
GT.add_TLS(is_qubit = False,
            TLS_label = 'defect1',
            energy = 4,
            couplings = {'qubit': [(0.08, [('sigmax', 'sigmax')])]
                        },
            Ls = {
                  #'sigmaz' : 0.005,
                  #'sigmay' : 0.002
                  }
            )
# GT.add_TLS(is_qubit = False,
#             energy = 4.5,
#             couplings = {'defect1': [(0.6, [('sigmap', 'sigmam'), ('sigmam', 'sigmap')]), 
#                                      (0.7, [('sigmax', 'sigmax')])]
#                         },
#             Ls = {
#                   'sigmaz' : 0.05,
#                   'sigmay' : 0.02
#                   }
#             )

GT.build_operators()

# simulate measurements:
# note: now using 1st qubit excited population at times ts
ts = np.logspace(0, 3, int(10))
# diffs = [ts[i] - ts[i-1] for i in range(len(ts)) if i>=1]
# #print(ts)
# print(diffs)
# for i in range(len(ts)):
#     ts[i] = ts[i] + (i/len(ts))**2
# #print(ts)
# diffs = [ts[i] - ts[i-1] for i in range(len(ts)) if i>=1]
# print(diffs)
#%%
measurement_observables = ['sigmax']
measurement_datasets = GT.calculate_dynamics(xs, observable_ops = measurement_observables)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(xs, measurement_datasets[0])



#%% perform learning:


# instance of learning (quest for best model):
quest = learning_chain.LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial = (5, 2), # (qubit energy, number of defects)
                      
                      max_chain_steps = 1000,
                      chain_step_options = {
                          'tweak all parameters': 0.5,
                          'add L': 0.05,
                          'remove L': 0.05,
                          'add qubit-defect coupling': 0.05, 
                          'remove qubit-defect coupling': 0.05,
                          'add defect-defect coupling': 0.025, 
                          'remove defect-defect coupling': 0.025
                          },
                      
                      temperature_proposal = 0.0001, # value or (shape, scale) to sample from gamma
                      
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
best = quest.run(3000)

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
    graphs = True # plot model graphs with corresponding labels
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
       models_to_save = [GT, best],
       model_names = ['GT', 'best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


#%% 




