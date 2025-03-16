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

    
# set up ground truth model:  
GT = basic_model.BasicModel()
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
            TLS_label = 'defect1',
            energy = 5.8,
            couplings = {'qubit': [(0.6, [('sigmay', 'sigmay')])]
                        },
            Ls = {
                  'sigmaz' : 0.05,
                  'sigmay' : 0.02
                  }
            )
GT.add_TLS(is_qubit = False,
            energy = 5.8,
            couplings = {'defect1': [(0.6, [('sigmaz', 'sigmaz')]), 
                                     (0.7, [('sigmax', 'sigmax')]),
                                     (0.8, [('sigmay', 'sigmay')])]
                        },
            Ls = {
                  'sigmaz' : 0.05,
                  'sigmay' : 0.02
                  }
            )

GT.build_operators()

# simulate measurements:
# note: now using 1st qubit excited population at times ts
ts = np.linspace(0, 1e1, int(1000))
measurement_observables = ['sigmaz']
measurement_datasets = GT.calculate_dynamics(ts, observable_ops = measurement_observables)

GT.disp()

#%% parallelised runs:


chains_inputs = []

def execute_chain():
    
    pass


# run chains in parallel:      
        
if __name__ == '__main__' and True:
    

    with multiprocessing.Pool() as pool:

        
        # run parallel processes

        chains_outputs = pool.map(execute_chain, chains_inputs)        
        
        


#%% perform learning:


# instance of learning (quest for best model):
quest = learning_chain.LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial = (5, 2), # (qubit energy, number of defects)
                      
                      max_chain_steps = 100,
                      chain_step_options = {
                          'tweak all parameters': 0.1,
                          'add L': 0.1,
                          'remove L': 0.1,
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
best = quest.run()
raise SystemExit()



costs = quest.explored_costs
acceptance_ratios = quest.acceptance_ratios_log
best_data = best.calculate_dynamics(ts, observable_ops = measurement_observables)



#%% save single learning run outputs:

    
# controls bundle:
    
class Toggles():    
    comparison = True # plot comparison of dynamics
    cost = True # plot cost function progression
    acceptance_ratios = True # plot acceptance ratios over subsequenct windows
    graphs = True # plot model graphs with corresponding labels
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True # save chain hyperparameters as json
    
    

# unique name (date and time stamp):

timestamp = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())


# create outputs:

output.Output(toggles = Toggles, filename = timestamp,
       dynamics_ts = ts, dynamics_datasets = [measurement_datasets[-1], best_data[-1]], dynamics_labels = ['ground truth', 'learned model'],
       cost = costs,
       acceptance_ratios = acceptance_ratios,
       models_to_save = [GT, best],
       model_names = ['GT', 'best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


#%% 



