#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


from learning_model import LearningModel

from params_optimiser import ParamsOptimiser

from model_modifier import ModelModifier

import numpy as np

from copy import deepcopy

import time



# single instance executes a learning chain (parameter space walk) by controlling learning model
# has methods for saving different instances of models
# controls hyperparameters for single chain

class LearningChain():
    
    
    
    
    
    
    def __init__(self, target_times, target_data, *,
                 initial_guess = False,              
                 params_optimiser_hyperparams = {'max_steps': int(1e3), 
                                                 'MH_acceptance': False, 
                                                 'MH_temperature': 10, 
                                                 'initial_jump_lengths': {'couplings' : 0.001,
                                                                          'energy' : 0.01,
                                                                          'Ls' : 0.00001
                                                                          }, 
                                                 'jump_annealing_rate': 0,
                                                 'acceptance_window': 200,
                                                 'acceptance_ratio': 0.4
                                                 },
                 model_modifier_process_library = {
                                                    'sigmax': 0.01
                                                   ,'sigmay': 0.01
                                                   #,'sigmaz': 0.01
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
        
        
        
        # model modifier object:
        # (ModelModifier instance)
        
        self.model_modification = None
        
        self.model_modifier_process_library = model_modifier_process_library 
        
        
        
        # initial guess model:
        
        if type(initial_guess) == bool and not initial_guess:
            
            self.initial = self.make_initial_guess()
        
        else:
            
            self.initial = initial_guess
            
            
            
        # target dataset:    
            
        self.target_data = target_data
        
        self.target_times = target_times
        
        
        
        
            
        
        
        
    # carry out Tiers 1, 2, 3 -- ie. proposing new TLSs, operators, and optimising parameters:    
        
    def learn(self):
        
        
        
        # initialise:
        
        self.explored_models = [] # repository of explored model configurations (now given by processes)
        
        self.explored_costs = []
        
        self.current = deepcopy(self.initial)
        
        max_modifications = 10
        
        
        
        # iteratively propose modifications and optimise parameters up to max_modifications times
        
        for i in range(max_modifications):
            

            # optimise current model with fixed structure
            
            self.explored_costs.append(self.optimise_params(self.current))
            
            self.explored_models.append(self.current)
                        
            
            # propose new model structure:     
            
            if i >= max_modifications - 1: break

            self.modify_model(self.current)
        
            
        
        best_index = self.explored_costs.index(min(self.explored_costs))
    
        return self.explored_models[best_index]
    
    
    
    
    
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
    
     
    
    
    # performs paramter optimisation on argument model, setting hyperparameterd to chain attribute,
    # saving resulting model to current working model and saving full cost progression and best cost achieved 
    # returns best cost achieved
    
    def optimise_params(self, model_to_optimise):
        
        if not self.params_optimisation: # ie. first run
            
            self.params_optimisation = ParamsOptimiser(self)
            
        self.params_optimisation.set_hyperparams(self.initial_params_optimiser_hyperparams)
        
        self.current, best_cost, costs = self.params_optimisation.do_optimisation(model_to_optimise)
        
        self.costs_brief.append(best_cost) 
        
        self.costs_full = self.costs_full + costs 
        
        return best_cost
        
        
        
        
    
    # returns JSON compatible dictionary of hyperparameters (relevant heuristics):
    # namely: initial jump lengths, annealing rate
    
    def chain_hyperparams_dict(self):
       
        chain_hyperparams_dict = {
                                  'initial guess': self.initial.model_description_dict()          
                                  }
        
        if self.params_optimisation:
            
            chain_hyperparams_dict['params optimisation initial hyperparameters'] = self.params_optimisation.output_hyperparams_init()
        
        return chain_hyperparams_dict 
    
    
    
    
    # performs model modification:
        
    def modify_model(self, model_to_modify):
        
        if not self.model_modification: # ie. first run
        
            self.model_modification = ModelModifier(self)
        
        self.model_modification.add_toss_L(model_to_modify, self.model_modifier_process_library)
    
    
    
    # 