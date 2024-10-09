#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


from learning_model import LearningModel

from params_optimiser import ParamsOptimiser

import numpy as np

from copy import deepcopy

import time



# single instance executes a learning chain (parameter space walk) by controlling learning model
# has methods for saving different instances of models
# controls hyperparameters for single chain

class LearningChain():
    
    
    
    
    
    
    def __init__(self, target_data, target_times, *,
              initial_guess = False,              
              # optimise_params_max_iter = False, # add plateau detection hyperparameters
              # jump_lengths = False,
              # jump_annealing_rate = 0,
              # MH_acceptance = False,
              # MH_temperature = 1
              ):
        
        
        
        
        # holders for different model instances in play:
        
        self.best = None
        
        self.current = None
        
        self.proposed = None
        
        
        
        # initial guess model:
        
        if type(initial_guess) == bool and not initial_guess:
            
            self.initial = self.make_initial_guess()
        
        else:
            
            self.initial = initial_guess
            
            
            
        # target dataset:    
            
        self.target_data = target_data
        
        self.target_times = target_times
        
        
        
        # optimise_params setting:
        
        
        # jump length and its annealing:   
        
        
        
        # if type(jump_lengths) == bool and not jump_lengths:
            
        #     self.jump_lengths = self.default_jump_lengths
            
        # else:
            
        #     self.jump_lengths = jump_lengths
        
        # self.initial_jump_lengths = deepcopy(self.jump_lengths)
        
        # self.jump_annealing_rate = jump_annealing_rate
        
        
        # # Metropolis-Hastings:
            
        # self.MH_acceptance = MH_acceptance
        
        # self.MH_temperature = MH_temperature
        
        
        
        # # maximum iterations for parameters optimisation setting:
            
        # self.default_optimise_param_max_iter = int(1e3)
        
        # if type(optimise_params_max_iter) == bool and not optimise_params_max_iter:
            
        #     self.optimise_params_max_iter = self.default_optimise_param_max_iter
            
        # else:
            
        #     self.optimise_params_max_iter = optimise_params_max_iter
            
            
        
            
        
        
        
    # here will go all the tiers:    
        
    def learn(self):    
        
        costs = []
                
        costs = costs + self.optimise_params()
        
        return costs
             
        
        
    
    
    
    
    
    def make_initial_guess(self):
        
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
                             energy = 4.5,
                             couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
                                          },
                             Ls = {
                                   'sigmaz' : 0.01
                                   }
                             )
        
        initial_guess.build_operators()
        
        return initial_guess
    
    
    

    # calculates and returns cost of model,
    # using target_times and target_data set at instance level,
    # currently using mean squared error between dynamics:
    # note: here is where any weighting or similar should be implemented
    
    def cost(self, model):
        
        
        model_data = model.calculate_dynamics(self.target_times)
        
        
        # check these are numpy arrays to use array operators below:
   
        if type(model_data) != type(np.array([])) or type(self.target_data) != type(np.array([])):
   
            raise RuntimeError('error calculating cost: arguments must be numpy arrays!\n')
       
        # print(model_data)
        # print(self.target_data)
        # print(abs(model_data-self.target_data))
        # print(np.sum(np.square(abs(model_data-self.target_data)))/len(self.target_times))
        
        return np.sum(np.square(abs(model_data-self.target_data)))/len(self.target_times)

     
    

    def optimise_params(self, model_to_optimise = False):
        
        params_optimisation = ParamsOptimiser(self)
        
        params_optimisation.do_optimisation()
        
    
        
    
    # returns JSON compatible dictionary of hyperparameters (relevant heuristics):
    # namely: initial jump lengths, annealing rate
    
    def chain_hyperparams_dict(self):
       
        return {'initial jump lengths': self.initial_jump_lengths,
                 'jump annealing rate': self.jump_annealing_rate,
                'M-H acceptance': self.MH_acceptance,
                 'M-H temperature': self.MH_temperature,
                 'initial guess': self.initial.model_description_dict()            
                 }
        
        pass
    
# also need to redefine hyperparams output!!



        