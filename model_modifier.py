#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import numpy as np

from model import Model

from learning_model import LearningModel


class ModelModifier():
    
    
    
    def __init__(self, chain, model = False, process_library = False):


        self.process_library = process_library   # dictionary: {'op label': initial_rate}
        
        self.model = model
        
        self.chain = chain

        

    
    def add_toss_L(self, model = False, process_library = False):
        
        
        
        # set model and process library if passed now, then check both set to appropriate types:
        
        if isinstance(model, LearningModel) or isinstance(model, Model):
            
            self.model = model
            
        if isinstance(process_library, dict):
            
            self.process_library = process_library
            
        if not (isinstance(self.process_library, dict)):
            
            raise SystemError('Model modification failed due to process library missing')
            
        if not (isinstance(model, LearningModel) or isinstance(model, Model)):
        
            raise SystemError('Model modification failed due to model not specified')
            
            
        
        # decide which system and whether to add or remove
        # note: for now acts on defects' Ls only
        
        subsystems = [x for x in self.model.TLSs if not x.is_qubit]
        
        subsystem = np.random.choice(subsystems)
        
        adding_of_L_probability = 0.7 # probability this step adds new process as opposed to removing
        
        if np.random.uniform() < adding_of_L_probability: # ie, add new process
        
            # randomly choose one process library operator not yet acting on chosen subsystem
        
            existing = [x for x in subsystem.Ls]
        
            options = [x for x in self.process_library if x not in existing]
                
            if options:
            
                op = np.random.choice(options)
                
                subsystem.Ls[op] = self.process_library[op]
                
                # note: this directly modifies variable another classe's object
                # ...perhaps could be changed to instead work via method of that class?
                
                self.model.build_operators()
                
                print('adding operator ' + op + ' to TLS no. ' + str(self.model.TLSs.index(subsystem)))
                
            else:
                
                print('no options for adding operator to TLS no. ' + str(self.model.TLSs.index(subsystem)))
            
        else: # ie, remove existing process
        
            options = [x for x in subsystem.Ls]
            
            if options:
                
                op = np.random.choice(options)
            
                subsystem.Ls.pop(op)
                
                print('removing operator ' + op + ' from TLS no. ' + str(self.model.TLSs.index(subsystem)))
                
            else:
                
                print('no options for removing operator from TLS no. ' + str(self.model.TLSs.index(subsystem)))
            
            
            
        # ADD check if process already present -- then choose another -- check also if options exhausted
        
        # or why not just choose difference... 
            
        
            
            

        
        