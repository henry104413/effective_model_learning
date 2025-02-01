#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

from model import Model

from learning_model import LearningModel

from learning_chain import LearningChain

from output import Output, compare_qutip_Liouvillian, create_model_graph

import multiprocessing

import numpy as np

import time


from definitions import Constants

import matplotlib.pyplot as plt







#%% generate target data:
    


# set up ground truth model:  

GT = Model()

GT.add_TLS(TLS_label = 'qubit',
           is_qubit = True,
           energy = 5,
           couplings = {
               
                        },
           Ls = {
                 'sigmaz' : 0.01
                 }
           )

GT.add_TLS(is_qubit = False,
            energy = 5.8,
            couplings = {'qubit': [(0.6, 'sigmax', 'sigmax')]
                        },
            Ls = {
                  'sigmaz' : 0.05,
                  'sigmay' : 0.02
                  }
            )


GT.add_TLS(is_qubit = False,
            energy = 4.5,
            couplings = {'qubit': [(0.3, 'sigmax', 'sigmax')]
                        },
            Ls = {
                  'sigmaz' : 0.03,
                  'sigmax' : 0.06
                  }
            )


# GT.add_TLS(is_qubit = False,
#             energy = 5.5,
#             couplings = {'qubit': [(0.2, 'sigmax', 'sigmax')]
#                         },
#             Ls = {
#                   'sigmaz' : 0.03
#                   }
#             )


# GT.add_TLS(is_qubit = False,
#             energy = 4.0,
#             couplings = {'qubit': [(0.4, 'sigmax', 'sigmax')]
#                         },
#             Ls = {
#                   'sigmaz' : 0.03
#                   }
#             )


# GT.add_TLS(is_qubit = False,
#             energy = 4.5,
#             couplings = {'qubit': [(0.1, 'sigmax', 'sigmax')]
#                         },
#             Ls = {
#                   'sigmaz' : 0.03
#                   }
#             )
     
GT.build_operators()


# simulate measurements:
# note: now using 1st qubit excited population at times ts

ts = np.linspace(0, 1e1, int(1000))

measurement_observables = ['sigmaz']

measurement_datasets = GT.calculate_dynamics(ts, observable_ops = measurement_observables)

# create_model_graph(GT, 'GTgraph')


#%% parallelised runs:


chains_inputs = []

def execute_chain():
    
    pass


# run chains in parallel:      
        
if __name__ == '__main__' and True:
    

    with multiprocessing.Pool() as pool:

        
        # run parallel processes

        chains_outputs = pool.map(execute_chain, chains_inputs)        
        
        
      

#%% chain hyperparameters:


#%% perform learning:


# instance of learning (quest for best model):
quest = LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial = (5, 2), # (qubit energy, number of defects)
                      
                      max_chain_steps = 100000,
                      chain_MH_temperature = 0.00001,
                      chain_MH_temperature_multiplier = 2,
                      chain_step_options = ['tweak all parameters',
                                            'add L', 'remove L',
                                            'add qubit coupling', 'remove qubit coupling',
                                            'add defect-defect coupling', 'remove defect-defect coupling'],
                      
                      chain_step_probabilities = [10,
                                                  1, 1, 
                                                  1, 1, 
                                                  1, 1],
                      
                      acceptance_window = 50,
                      acceptance_target = 0.4,
                      acceptance_band = 0.2,
                      
                      params_handler_hyperparams = { 
                          'initial_jump_lengths': {'couplings' : 0.01,
                                                   'energy' : 0.1,
                                                   'Ls' : 0.001
                                                   },
                          },
                      
                      Ls_library = { # will draw from uniform distribution from specified range)
                                                         'sigmax': (0.01, 0.1)
                                                        ,'sigmay': (0.01, 0.1)
                                                        ,'sigmaz': (0.01, 0.1)
                                                        },
           
                      qubit_couplings_library = { # will draw from uniform distribution from specified range)
                          'sigmax': (-1, 1)
                         ,'sigmay': (-1, 1)
                         ,'sigmaz': (-1, 1)
                         },
                      
                      defect_couplings_library = { # will draw from uniform distribution from specified range)
                          'sigmax': (-1, 1)
                         ,'sigmay': (-1, 1)
                         ,'sigmaz': (-1, 1)
                         }

                      
                      )

#%%

# import params_handling

# ph = params_handling.ParamsHandler(quest)

# ph.tweak_all_parameters(initial_guess)


best = quest.learn()

#raise SystemExit(0)

costs = quest.explored_costs
acceptance_ratios = quest.acceptance_ratios_log

best_data = best.calculate_dynamics(ts, observable_ops = measurement_observables)

#%%

create_model_graph(GT, 'GT_graph')


create_model_graph(best, 'best_graph')



#%% save single learning run outputs:

    
# controls bundle:
    
class Toggles():    
    comparison = True # plot comparison of dynamics
    cost = True # plot cost function progression
    acceptance_ratios = True # acceptance ratios over subsequenct windows
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True # save chain hyperparameters as json
    
    

# unique name (date and time stamp):

timestamp = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())


# create outputs:

Output(toggles = Toggles, filename = timestamp,
       dynamics_ts = ts, dynamics_datasets = [measurement_datasets[-1], best_data[-1]], dynamics_labels = ['ground truth', 'learned model'],
       cost = costs,
       acceptance_ratios = acceptance_ratios,
       models_to_save = [GT, best],
       model_names = ['GT', 'best'],
       chain_hyperparams = quest.chain_hyperparams_dict()
       )


#%% 



