#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


from learning_model import LearningModel

import numpy as np

from copy import deepcopy

import time



# single instance executes a learning chain (parameter space walk) by controlling learning model
# has methods for saving different instances of models
# controls hyperparameters for single chain

class LearningChain():
    
    
    
    
    
    
    def __init__(self, target_data, target_times, *,
              initial_guess = False,              
              optimise_params_max_iter = False, # add plateau detection hyperparameters
              jump_lengths = False,
              jump_annealing_rate = 0,
              MH_acceptance = False
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
        
        self.default_jump_lengths = {
                                'couplings' : 0.001,
                                'energy' : 0.01,
                                'Ls' : 0.00001
                                }
        
        if type(jump_lengths) == bool and not jump_lengths:
            
            self.jump_lengths = self.default_jump_lengths
            
        else:
            
            self.jump_lengths = jump_lengths
        
        self.initial_jump_lengths = deepcopy(self.jump_lengths)
        
        self.jump_annealing_rate = jump_annealing_rate
        
        
        # Metropolis-Hastings:
            
        self.MH_acceptance = MH_acceptance
        
        
        
        # maximum iterations for parameters optimisation setting:
            
        self.default_optimise_param_max_iter = int(1e3)
        
        if type(optimise_params_max_iter) == bool and not optimise_params_max_iter:
            
            self.optimise_params_max_iter = self.default_optimise_param_max_iter
            
        else:
            
            self.optimise_params_max_iter = optimise_params_max_iter
            
            
            
        # profiling - total time tracking variables:
            
        self.profiling_optimise_params_manipulations = 0.0
        self.profiling_optimise_params_cost_eval = 0.0 
        
            
        
        
        
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
        
        
    
        # initialise:
    
        if not model_to_optimise: self.current = self.initial
       
        else: self.current = model_to_optimise
        
        current_cost = self.cost(self.current)
        
        costs = []
        
        costs.append(current_cost)
        
        best_cost = current_cost
        
        self.rescale_jump_lengths(10)
        
        
        
        for i in range(self.optimise_params_max_iter): # add plateau condition
            
            
            # profiling timer:
                
            clock = time.time()
            
            
            # make copy of model, propose new parameters and evaluate cost:
            
            self.proposed = deepcopy(self.current) 
        
            self.proposed.change_params(passed_jump_lengths = self.jump_lengths)
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
                
            clock = time.time()
            
            proposed_cost = self.cost(self.proposed)
            
            self.profiling_optimise_params_cost_eval += (time.time() - clock)
            
            clock = time.time()
            
            costs.append(proposed_cost)
            
            
            
            # anneal jump length:
                
            if self.jump_annealing_rate:
                
                self.rescale_jump_lengths(np.exp(-self.jump_annealing_rate))
            
            
            
            # if improvement -- accept and update current, check if best and save then:
            
            if proposed_cost < current_cost: 
                
                self.current = self.proposed 
                
                current_cost = proposed_cost
                
                if proposed_cost < best_cost:
                    
                    best_cost = proposed_cost
                    
                    self.best = deepcopy(self.proposed)
                
                
                # actually maybe assume that best is always current UNLESS doing MH acceptance!!!!
                
                #ALSO NOPE!!!!
                
            # if detriment, still accept with given likelyhood (like in Metropolis-Hastings)
            # AND update best to current AND update current to proposal (now worse than previous current)
            
            elif (self.MH_acceptance 
                  and np.exp(proposed_cost - current_cost)*self.MH_temperature > np.random.uniform()): 
                
                print('using MH:\nold cost: ' + str(current_cost) + '\nnew cost: ' + str(proposed_cost))
                
                self.current = self.proposed 
                
                current_cost = proposed_cost
                
            
            # else reject: (proposal will be discarded and fresh one made from current at start of loop)
            else:
               
                pass # proposed will be discarded and a fresh one made from current
                
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
                
        
        
        return costs
    
        # to do: maybe the best model should also be specific to the run of optimise_params (could do more at once)
                
        
        
        # by the way:
        # only update best at the end (to current) or when taking a worse step - metropolis hastings
        # would require knowing in advance when things will end though... or keeping two best models...
        # in any case probably don't have to worry - it's a small expense
        
    
    
    # returns JSON compatible dictionary of hyperparameters (relevant heuristics):
    # namely: initial jump lengths, annealing rate
    
    def chain_hyperparams_dict(self):
       
        return {'initial jump lengths': self.initial_jump_lengths,
                'jump annealing rate': self.jump_annealing_rate,
                'initial guess': self.initial.model_description_dict()            
                }
        
    


    # rescales all jump lengths by given factors: (also used for annealing)
    
    def rescale_jump_lengths(self, factor):
        
        for key in self.jump_lengths:
            
            self.jump_lengths[key] = self.jump_lengths[key]*factor
       