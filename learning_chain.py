#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:01:39 2024

@author: henry
"""

from learning_model import LearningModel

import numpy as np



# single instance executes a learning chain (parameter space walk) by controlling learning model
# has methods for saving different instances of models

class LearningChain():
    
    
    
    
    
    
    def __init__(self, target_data, target_times,
              initial_guess = False,              
              method_config = False, # can turn this into multiple arguments
              ):
        
        
        
        
        # holders for different model instances in play:
        
        self.best = False
        
        self.current = False
        
        
        # initial guess model:
        
        if type(initial_guess) == bool and not initial_guess:
            
            self.initial = self.make_initial_guess()
        
        else:
            
            self.initial = initial_guess
            
            
        self.target_data = target_data
        
        self.target_times = target_times
        
        
        
    # first take on learning:    
        
    def learn(self):    
        
        pass
    
    
    
    def optimise_parameters(self):
        
        pass
    
    
    def save_best(self):
        
        pass    
    
    
    
    
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
                             energy = 4,
                             couplings = {'qubit': [(0.4, 'sigmax', 'sigmax')]
                                          },
                             Ls = {
                                   'sigmaz' : 0.01
                                   }
                             )
        
        return initial_guess
    
    

    # calculates cost of model, using target_times and target_data set at instance level,
    # currently using mean squared error between dynamics:
    # note: here is where any weighting or similar should be implemented
    
    def cost(self, model):
        
        
        model_data = model.calculate_dynamics(self.target_times)
        
        
        # check these are numpy arrays to use array operators below:
   
        if type(model_data) != type(np.array([])) or type(self.gtarget_data) != type(np.array([])):
   
            raise RuntimeError('error calculating cost: arguments must be numpy arrays!\n')
       
        
        return np.sum(np.square(abs(model_data-self.target_data)))/len(self.target_times)

        
        