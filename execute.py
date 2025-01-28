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
                 'sigmaz' : 0.005
                 }
           )

GT.add_TLS(is_qubit = False,
            energy = 5.5,
            couplings = {'qubit': [(0.6, 'sigmax', 'sigmax')]
                        },
            Ls = {
                  'sigmaz' : 0.02,
                  'sigmay' : 0.02
                  }
            )


GT.add_TLS(is_qubit = False,
            energy = 4.5,
            couplings = {'qubit': [(0.3, 'sigmax', 'sigmax')]
                        },
            Ls = {
                  'sigmaz' : 0.03,
                  'sigmax' : 0.03
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

measurement_observables = ['sigmax']

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
# now change this to having an iterable for parallelisation
    
    
# initial guesss generation:

initial_guess = LearningModel()


initial_guess.add_TLS(TLS_label = 'qubit',
                     is_qubit = True,
                     energy = 5,
                     couplings = {
                                  
                                  },
                     Ls = {
                           #'sigmaz' : 0.01
                           }
                     )


initial_guess.add_TLS(is_qubit = False,
                     energy = 5.0,
                     couplings = {#'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           #'sigmax' : 0.01,
                           #'sigmay' : 0.01
                           
                           }
                     )


initial_guess.add_TLS(is_qubit = False,
                     energy = 5.0,
                     couplings = {#'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           #'sigmax' : 0.01,
                           #'sigmay' : 0.01
                           
                           }
                     )


# initial_guess.add_TLS(is_qubit = False,
#                       energy = 5.0,
#                       couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
#                                   },
#                       Ls = {
#                             #'sigmax' : 0.01,
#                             #'sigmay' : 0.01
#                             }
#                       )


# initial_guess.add_TLS(is_qubit = False,
#                      energy = 5.0,
#                      couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
#                                   },
#                      Ls = {
#                            'sigmaz' : 0.01
#                            }
#                      )





initial_guess.build_operators()




#%% perform learning:


# instance of learning (quest for best model):
quest = LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial_guess = initial_guess,
                      
                      max_chain_steps = 1000,
                      chain_MH_temperature = 0.00001,
                      chain_step_options = ['tweak all parameters', 'add L', 'remove L',
                                            'add qubit coupling', 'remove qubit coupling',
                                            'add defect-defect coupling', 'remove defect-defect coupling'],
                      # chain_step_probabilities = [10,
                      #                             0.1, 0.1, 
                      #                             0.05, 0.05, 
                      #                             0.02, 0.02],
                      chain_step_probabilities = [10,
                                                  2, 1, 
                                                  2, 1, 
                                                  2, 1],
                      
                      acceptance_window = 100,
                      acceptance_target = 0.4,
                      
                      params_handler_hyperparams = { 
                          'initial_jump_lengths': {'couplings' : 0.001,
                                                   'energy' : 0.01,
                                                   'Ls' : 0.00001
                                                   },
                          },
                      
                      Ls_library = { # will draw from uniform distribution from specified range)
                                                         'sigmax': (0.001, 0.1)
                                                        ,'sigmay': (0.001, 0.1)
                                                        ,'sigmaz': (0.001, 0.1)
                                                        }
                      
                      )

#%%

# import params_handling

# ph = params_handling.ParamsHandler(quest)

# ph.tweak_all_parameters(initial_guess)


best = quest.learn()

#raise SystemExit(0)

costs = quest.explored_costs

best_data = best.calculate_dynamics(ts, observable_ops = measurement_observables)

#%%

create_model_graph(GT, 'GT_graph')


create_model_graph(best, 'best_graph')


create_model_graph(initial_guess, 'initial_guess')



#%% save single learning run outputs:

    
# controls bundle:
    
class Toggles():    
    comparison = True # plot comparison of dynamics
    cost = True # plot cost function progression
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True # save chain hyperparameters as json
    
    

# unique name (date and time stamp):

timestamp = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())


# create outputs:

Output(toggles = Toggles, filename = timestamp,
       dynamics_ts = ts, dynamics_datasets = [measurement_datasets[-1], best_data[-1]], dynamics_labels = ['ground truth', 'learned model'],
       cost = costs,
       models_to_save = [GT, best],
       model_names = ['GT', 'best'],
       chain_hyperparams = quest.chain_hyperparams_dict()
       )


#%% 



