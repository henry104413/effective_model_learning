#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:01:39 2024

@author: henry
"""

from learning_model import LearningModel



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
        
        
        if type(initial_guess) == bool and not initial_guess:
            
            self.initial = self.make_initial_guess()
        
        else:
            
            self.initial = initial_guess
        
        
        pass
    
    
    
    def optimise_parameters(self):
        
        pass
    
    
    def save_best(self):
        
        pass    
    
    def make_initial_guess():
        
        
        
        pass
    
    
    