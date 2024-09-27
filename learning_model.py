#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


from model import Model

import numpy as np


# enhanced model with ability to modify itself
# methods to change parameters, methods to add/remove processes, and methods to add/remove subsystems

class LearningModel(Model):
    
    
    
    def __init__(self, *args, initial_guess = False, jump_lengths = False, **kwargs):
        
        
        
        # parent constructor sets up model parameters:
        # note: for now models initialised empty and systems added dynamically 
        
        super().__init__(*args, **kwargs)
        
        
        
        # dictionary of standard jump lengths for each type of operator
        # note: modified with direct method or with argument to parameter changing method
        
        self.jump_lengths = jump_lengths
        
        
        
    
    def set_jump_lengths(self, passed_jump_lengths):
        
        self.jump_lengths = passed_jump_lengths
        
        
    
    
    def change_params(self, passed_jump_lengths = False):
        
        
        # check jump lengths specified:
            
        if (type(passed_jump_lengths) == bool and not passed_jump_lengths):
            
            if (type(self.jump_lengths) == bool and not self.jump_lengths):
            
                raise RuntimeError('need to specify jump lengths for the operators present')
        
        else:
            
            self.jump_lengths = passed_jump_lengths
               
        
        # execute jump in arbitrary direction, ie change all parameters:
            
        # go over all TLSs:
            
        for TLS in self.TLSs:
                
                    
            # modify energy but not on qubit:
                
            if not TLS.is_qubit:
            
                TLS.energy += np.random.normal(0, self.jump_lengths['energy'])
            
            
            # couplings:
            
            for partner in TLS.couplings: # partner is key and value is list of tuples
                
                # disabled - this would be good if the coupling tuple were a mutable array:
                # for coupling in TLS.couplings[partner]: # coupling is a 3-tuple from the list
                    
                #     coupling = (coupling[0] + random.normal(0, self.jump_lengths['couplings']),
                #                 coupling[1], coupling[2]) # note: this reassigns the tuple so not sure if efficient
                
                current_list = TLS.couplings[partner] # list of couplings to current partner
                
                for i in range(len(current_list)):
                    
                    current_list[i] = (current_list[i][0] + np.random.normal(0, self.jump_lengths['couplings']),
                                       current_list[i][1], current_list[i][2])
                    
                    
            
            # Lindblad ops:
                
            for L in TLS.Ls:
                
                
                # make up to specified number of proposals ensuring result positive
                    
                max_attempts = 10
                
                for _ in range(max_attempts):    
                    
                    proposal = TLS.Ls[L] + np.random.normal(0, self.jump_lengths['Ls'])
                    
                    if proposal >= 0:
                        
                        TLS.Ls[L] = proposal
                                
                        break
                        
                        
                                        
                
                    
                    
                    
                    
        # remake operators:
            
        self.build_operators()
        
                
                
                
        
            
        
        
        