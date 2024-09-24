#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

from model import Model

from learning_model import LearningModel

from learning_chain import LearningChain

from definitions import Constants

import numpy as np

import matplotlib.pyplot as plt

import time

import pickle



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

costs = quest.learn(100000)

best = quest.best

best_data = best.calculate_dynamics(ts)




#%% save single learning run outputs:

    
# controls bundle:
    
class Save():    
    comparison = True
    cost = True
    GT_model = True
    best_model = True
    GT_text = True
    best_text = True



# unique name (date and time stamp):

timestamp = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())



# plot dynamics comparison:

plt.figure()
plt.plot(ts*Constants.t_to_sec, measurements, 'r-', label = 'ground truth')
plt.plot(ts*Constants.t_to_sec, best_data, 'b--', label = 'learned model')
plt.xlabel('time (fs)')
plt.ylabel('qubit excited population')
plt.ylim([0,1.1])
plt.legend()
if Save.comparison: plt.savefig(timestamp + '_comparison.svg')



# plot cost function progression:
    
plt.figure()
plt.plot(costs, 'm-', linewidth = 0.1, markersize = 0.1)
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('cost')
if Save.cost: plt.savefig(timestamp + '_cost.svg')



# save ground truth instance:

if Save.GT_model:
    with open(timestamp + '_GT.pickle', 'wb') as filestream:
        pickle.dump(GT,  filestream)
    
    
    
# save best model instance:

if Save.best_model:
    with open(timestamp + '_best.pickle', 'wb') as filestream:
        pickle.dump(best,  filestream)
    
    
    
# save ground truth text descritpion:

if Save.GT_text:
    with open(timestamp + '_GT.txt', 'w') as filestream:
        filestream.write(GT.description())
    


# save ground truth text descritpion:

if Save.GT_text:
    with open(timestamp + '_best.txt', 'w') as filestream:
        filestream.write(best.description())
    


    
    
    
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

