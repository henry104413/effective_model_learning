#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

class TwoLevelSystem():
    
    
    # makes new TLS instance:
    # arguments: model instance this belongs to; qubit indicator (false means defect); energy;
    # dictionary of coherent couplings where keys are other TLSs in this model; and values are tuples:
    # ...(rate, operator acting on this, operator acting on other TLS);
    # dictionary of linblad operators where key is operator and value is rate
    
    def __init__(self, model, TLS_id = "", is_qubit = False, energy = None, couplings = {}, Ls = {}):
        
        
        # to add: input check
        # inluding: operators defined, and structure of dictionaries... later though
        
        self.model = model
        
        self.TLS_id = TLS_id
             
        self.is_qubit = is_qubit
        
        self.energy = energy
        
        self.couplings = couplings
        
        
                 
        # future: call method updating coupling on partners but not done now        
        # this way each TLS would contain complete information about itself
        # Hamiltonian builder function should then make sure it only applies this once
        # operator builders should probably be model methods as the dimensionality is model dependent
        
        
        self.Ls = Ls
        
        
    
            
    
    
    # in future, this would update coupling information on partner as well
    # then each TLS will have full information about the couplings
    # but operator creating methods will have to be amended to avoid double counting!
    # for now probably unnecessary -- model contains full information, individual TLS objects don't really matter
    
    def update_couplings_on_partner(self, couplings):
        
        pass
                 
    
    
                 
                 
                 
                 
                 
                 
                 
                 
                 