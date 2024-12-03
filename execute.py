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
                 #'sigmaz' : 0.005
                 }
           )

GT.add_TLS(is_qubit = False,
           energy = 5.5,
           couplings = {'qubit': [(0.6, 'sigmax', 'sigmax')]
                        },
           Ls = {
                 #'sigmaz' : 0.02,
                 'sigmay' : 0.02,
                 }
           )


GT.add_TLS(is_qubit = False,
            energy = 4.5,
            couplings = {'qubit': [(0.3, 'sigmax', 'sigmax')]
                        },
            Ls = {
                  #'sigmaz' : 0.03
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

ts = np.linspace(0, 4e1, int(1000))

measurements = GT.calculate_dynamics(ts, dynamics_method = 'qutip')

# create_model_graph(GT, 'GTgraph')


#%% 


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
                     couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           #'sigmax' : 0.01,
                           #'sigmay' : 0.01
                           
                           }
                     )

initial_guess.add_TLS(is_qubit = False,
                      energy = 5.0,
                      couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                      Ls = {
                            #'sigmax' : 0.01,
                            #'sigmay' : 0.01
                            }
                      )


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
quest = LearningChain(target_times = ts, target_data = measurements,
                      initial_guess = initial_guess,
                      params_optimiser_hyperparams = {'max_steps': int(3000), 
                                                      'MH_acceptance': not True, 
                                                      'MH_temperature': 1e-4, # 1 means no change to criterion
                                                      # MH temp 1e-3 seems to give noral values for below jump lengths
                                                      # 'initial_jump_lengths': {'couplings' : 0.001,
                                                      #                          'energy' : 0.01,
                                                      #                          'Ls' : 0.00001
                                                      #                          }, 
                                                      'initial_jump_lengths': {'couplings' : 0.001,
                                                                               'energy' : 0.01,
                                                                               'Ls' : 0.00001
                                                                               }, 
                                                      'jump_annealing_rate': 0
                                                      }
                      )


best = quest.learn()

costs = quest.costs_full

best_data = best.calculate_dynamics(ts)

#%%

create_model_graph(GT, 'GT_graph')


create_model_graph(best, 'best_graph')




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
       dynamics_ts = ts, dynamics_datasets = [measurements, best_data], dynamics_labels = ['ground truth', 'learned model'],
       cost = costs,
       models_to_save = [GT, best],
       model_names = ['GT', 'best'],
       chain_hyperparams = quest.chain_hyperparams_dict()
       )


#%% 

"""
qutip occurences:
    
    in model class to evaluate dynamics
    
    in definitions to produce T and operators
    
    so just replace T and operators there
    
    and replace dynamics in model class with exponential solver



"""



