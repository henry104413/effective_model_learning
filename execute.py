#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

from model import Model

from learning_model import LearningModel

from learning_chain import LearningChain

from output import Output

import numpy as np

import time



#%% set up ground truth model:

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
           energy = 4.5,
           couplings = {'qubit': [(0.4, 'sigmax', 'sigmax')]
                        },
           Ls = {
                 'sigmaz' : 0.02
                 }
           )


GT.add_TLS(is_qubit = False,
           energy = 4.0,
           couplings = {'qubit': [(0.7, 'sigmax', 'sigmax')]
                        },
           Ls = {
                 'sigmaz' : 0.03
                 }
           )


GT.add_TLS(is_qubit = False,
           energy = 5.3,
           couplings = {'qubit': [(0.3, 'sigmax', 'sigmax')]
                        },
           Ls = {
                 'sigmaz' : 0.01
                 }
           )


        
GT.build_operators()



# generate data measurements:
# note: now using 1st qubit excited population at times ts

ts = np.linspace(0, 5e1, int(1000))

measurements = GT.calculate_dynamics(ts)




#%% initial guess:

initial_guess = LearningModel()


initial_guess.add_TLS(TLS_label = 'qubit',
                     is_qubit = True,
                     energy = 5,
                     couplings = {
                                  
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )


initial_guess.add_TLS(is_qubit = False,
                     energy = 5.0,
                     couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )

initial_guess.add_TLS(is_qubit = False,
                     energy = 5.0,
                     couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )


initial_guess.build_operators()




#%% perform learning:


# instance of learning (quest for best model):
quest = LearningChain(target_times = ts, target_data = measurements, initial_guess = initial_guess)

costs = quest.learn(int(3e5))

best = quest.best

best_data = best.calculate_dynamics(ts)




#%% save single learning run outputs:

    
# controls bundle:
    
class Toggles():    
    comparison = True # plot comparison of dynamics
    cost = True # plot cost function progression
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True 
    
    

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
    

#%% works!
# add accompanying document with quick description or something
# could be jason or really a txt will do


# next: wrap this into ouput class in another file
# add lindblad adding or removing step
# how to decide?
# multiple chain for stats and add paralelisation
# run on gauge

# add random initial guess
# run multiple chains
# see best outcome etc

# add central font etc

# add another file to output with hyperparameters - json or text - to keep track of how it was generated

# possibly profile
