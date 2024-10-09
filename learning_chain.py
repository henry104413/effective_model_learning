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
                 params_optimiser_hyperparams = {'max_steps': int(1e3), 
                                                 'MH_acceptance': False, 
                                                 'MH_temperature': 0.1, 
                                                 'initial_jump_lengths': {'couplings' : 0.001,
                                                                          'energy' : 0.01,
                                                                          'Ls' : 0.00001
                                                                          }, 
                                                 'jump_annealing_rate': 0
                                                 }
                 ):
        
        
        
        
        # holders for different model instances in play:
        
        self.best = None
        
        self.current = None
        
        
        
        # chain costs trackers:
            
        self.costs_full = [] # includes full cost progression from each parameters optimiser call
        
        self.costs_brief = [] # only includes best cost from each parameters optimiser call
        
        
        
        # optimiser object and initial hyperparameters for it:
        # (PramsOptimiser instance with attributes for hyperparams
        # ...and methods to carry out params optimisation, set and output hyperparams)
        
        self.params_optimisation = None 
        
        self.initial_params_optimiser_hyperparams = params_optimiser_hyperparams
        
        
        
        # initial guess model:
        
        if type(initial_guess) == bool and not initial_guess:
            
            self.initial = self.make_initial_guess()
        
        else:
            
            self.initial = initial_guess
            
            
            
        # target dataset:    
            
        self.target_data = target_data
        
        self.target_times = target_times
        
        
        
        
            
        
        
        
    # here will go all the tiers:    
        
    def learn(self):
        
        self.current = deepcopy(self.initial)
        
        self.optimise_params(self.current)     
        
        self.best = self.current
        
        return self.best
    
    
    
    
    
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
       
        
        return np.sum(np.square(abs(model_data-self.target_data)))/len(self.target_times)

     
    
    

    def optimise_params(self, model_to_optimise):
        
        if not self.params_optimisation: # ie. first run
            
            self.params_optimisation = ParamsOptimiser(self)
            
        self.params_optimisation.set_hyperparams(self.initial_params_optimiser_hyperparams)
        
        self.current, best_cost, costs = self.params_optimisation.do_optimisation(model_to_optimise)
        
        self.costs_brief.append(best_cost) 
        
        self.costs_full = self.costs_full + costs 
        
        
        
    
    # returns JSON compatible dictionary of hyperparameters (relevant heuristics):
    # namely: initial jump lengths, annealing rate
    
    def chain_hyperparams_dict(self):
       
        chain_hyperparams_dict = {
                                  'initial guess': self.initial.model_description_dict()          
                                  }
        
        if self.params_optimisation:
            
            chain_hyperparams_dict['params optimisation hyperparameters'] = self.params_optimisation.output_hyperparams()
      
        return chain_hyperparams_dict 
    


        