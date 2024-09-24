#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


from learning_model import LearningModel

import numpy as np

from copy import deepcopy



# single instance executes a learning chain (parameter space walk) by controlling learning model
# has methods for saving different instances of models

class LearningChain():
    
    
    
    
    
    
    def __init__(self, target_data, target_times,
              initial_guess = False,              
              method_config = False, # can turn this into multiple arguments
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
            
            
        self.target_data = target_data
        
        self.target_times = target_times
        
        
        
        
    # first take on learning:    
        
    def learn(self, iterations):    
        
        
        costs = []
                
        costs = costs + self.optimise_parameters(iterations)
        
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

        
        

    def optimise_parameters(self, iterations):
        
            self.current = self.initial
            
            current_cost = self.cost(self.current)
            
            costs = []
            
            costs.append(current_cost)
            
            
            
            for i in range(iterations):
                
                
                # make copy of model, propose new parameters and evaluate cost:
                
                self.proposed = deepcopy(self.current) 
            
                self.proposed.change_params() #
                
                proposed_cost = self.cost(self.proposed)
                
                costs.append(proposed_cost)
                
                
                # if improvement, accept and update current:
                
                if proposed_cost < current_cost: 
                    
                    self.current = self.proposed 
                    
                    current_cost = proposed_cost
                    
                
                # if detriment, still accept with given likelyhood (like in Metropolis-Hastings)
                # AND update best to current AND update current to proposal (now worse than previous current)
                
                elif False: 
                    
                    pass
                
                
                # else reject: (proposal will be discarded and fresh one made from current at start of loop)
                else:
                   
                    continue # proposed will be discarded and a fresh one made from current
                    
                   
            self.best = self.current   
            
            return costs
                    
            
            
            # by the way:
            # only update best at the end (to current) or when taking a worse step - metropolis hastings
            
        
    
   