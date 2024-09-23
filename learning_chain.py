#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:01:39 2024

@author: henry
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
        
        self.current = self.initial
        
        current_cost = self.cost(self.current)
        
        costs = []
        
        costs.append(current_cost)
        
        for i in range(iterations):
            
            
            # actually do loops until a step is accepted here
        
            self.proposed = deepcopy(self.current) # as if this doesn't get carried out...
        
            self.proposed.change_params() # and this neither
            
            proposed_cost = self.cost(self.proposed)
            
            costs.append(proposed_cost)
        
            ##
            print('...........................................')
            print('\n\n***Current***')
            self.current.print_params()
            print('\n\n***Proposed***')
            self.proposed.print_params()
            #
                
            if proposed_cost < current_cost: # ie improvement so accept
                
                self.current = self.proposed
                
                current_cost = proposed_cost
                
                print('\n\nACCEPTED\n')
            
            else:
               
                print('rejected')
               
                continue # quite an awkward step back this... need to deepcopy again
               
               
        self.best = self.current   
        
        return costs
                
        
        
        # by the way:
        # only update best at the end (to current) or when taking a worse step - metropolis hastings
        
        
        
    
    
    
    
    
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

        
        

    def optimise_parameters(self):
        
        pass
    
    
    def save_best(self):
        
        pass          
    
    
    # after first improvement on CURRENT there seems to be no change ever after in either direction!!!!
    # 