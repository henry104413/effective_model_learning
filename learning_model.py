#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:01:39 2024

@author: henry
"""

from model import Model

class LearningModel(Model):
    
    
    def learn(target_data, target_times,
              evaluation_times, 
              method = 'MHMCMC', 
              max_steps = 1000, 
              plateau_window = False, #(50, 10), 
              method_config = False, #{'normalised step size' = 1, '...' = 0}
              # note: this can constain all the details about what steps are allowed etc.
              # ...possibly package all these into an argument tuple
              ):
        
        
        pass