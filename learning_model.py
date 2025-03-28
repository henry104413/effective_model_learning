#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


import basic_model
import numpy as np


# enhanced model with ability to modify itself
# methods to change parameters, methods to add/remove processes, and methods to add/remove subsystems

class LearningModel(basic_model.BasicModel):
    
    """
    Instances based on BasicModel (hold information about systems, couplings, processes;
    able to generate full operators and dynamics accodring to Lindblad Master Equation).
    Additionally implements method to shift all parameters by randomly sampled amount,
    with width for each type of parameter set for instance as jump lengths hyperparameter.
    """    
    
    def __init__(self, *args, initial_guess = False, jump_lengths = False, **kwargs):
        
        # parent initialiser sets up model parameters:
        # note: often initialised empty and components added dynamically 
        super().__init__(*args, **kwargs)
        
        # dictionary of standard jump lengths for each type of operator
        # note: modified with direct method or with argument to parameter changing method
        self.jump_lengths = jump_lengths
        
        
    
    def set_jump_lengths(self, passed_jump_lengths: dict[str, float|int] = False) -> None:
        
        """
        Sets jump lengths according to argument dictionary.
        """
        
        self.jump_lengths = passed_jump_lengths
        
    
    
    def change_params(self, passed_jump_lengths: dict[str, float|int] = False) -> None:
        
        """
        Changes all existing parameters of this model except qubit energies,
        each by amount from normal distribution around zero,
        with variance given for each class of parameters (current split: energy, couplings, Ls)
        by jump lengths dictionary if passed or instance variable if not.
        """
        
        # check jump lengths specified:
        if (type(passed_jump_lengths) == bool and not passed_jump_lengths):
            if (type(self.jump_lengths) == bool and not self.jump_lengths):
                raise RuntimeError('need to specify jump lengths for the operators present')
        else:
            self.jump_lengths = passed_jump_lengths
               
        # take each TLS:    
        for TLS in self.TLSs:
                    
            # modify its energy if not qubit:
            if not TLS.is_qubit:
                TLS.energy += np.random.normal(0, self.jump_lengths['energies'])
            
            # modify all its couplings to each partner:
            # {partner: [(rate, [(op_on_self, op_on_partner)])]}
            for partner in TLS.couplings: # partner is key and value is list of tuples
                this_partner_couplings = TLS.couplings[partner] # list of couplings to current partner
                for i, coupling in enumerate(TLS.couplings[partner]): # coupling now (rate, [(op_on_self, op_on_partner), ...])
                    strength, op_pairs = coupling
                    TLS.couplings[partner][i] = (strength + np.random.normal(0, self.jump_lengths['couplings']), op_pairs)
            
            # modify all its Lindblad ops:
            for L in TLS.Ls:
                # make up to specified number of proposals ensuring result positive
                max_attempts = 10
                for _ in range(max_attempts):    
                    proposal = TLS.Ls[L] + np.random.normal(0, self.jump_lengths['Ls'])
                    if proposal >= 0:
                        TLS.Ls[L] = proposal
                        break
                 
        # remake operators (to update with new parameters):
        self.build_operators()
        
        
                
                        
            
        
        
        