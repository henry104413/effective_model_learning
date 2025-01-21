#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import numpy as np

from model import Model
from learning_model import LearningModel


# executes any modification to model processes (operators)
# stores available process library

# methods should take model as argument and work on it
# same process handling object expected to be used on various models 


class ProcessHandler():
    
    
    
    def __init__(self, chain, model = False, process_library = False):


        self.process_library = process_library   # dictionary: {'op label': (lower_bound, upper_bound)}
        
        self.model = model
        
        self.chain = chain

        

    # decides and carries out either addition or removal of a random L on a random defect
    # - currently unused 
    
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
        
        adding_of_L_probability = 0.5 # probability this step adds new process as opposed to removing
        
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
            
        
            
    # adds random new single-site Linblad process from process library to a random subsystem (qubit or defect for now) 
    # with rate drawn from uniform distribution between bounds given by tuple for each process library entry  
    # works on and returns argument model
    # argument process library used if passed, otherwise use that of handler if already set, error if neither available  
            
    def add_random_L(self, model, process_library = False):
        
        
        if process_library == False:
            
            # can either be passed as each call 
            
            if isinstance(self.process_library, dict):
                
                process_library = self.process_library
            
            else:
                
                raise RuntimeError('Cannot add Lindblad process due to process library not specified')
                
        
        
        # select subsystem:
        
        subsystems = [x for x in model.TLSs]# if not x.is_qubit]
        
        subsystem = np.random.choice(subsystems)
        
        
        # identify Ls in library not yet present on selected subsystem:
            
        existing = [x for x in subsystem.Ls]
    
        options = [x for x in process_library if x not in existing] 
        
  
        # select and add new L if available:
            
        if options:
        
            op = np.random.choice(options)
            
            # draw rate from uniform distribution between bounds: (note: need to unwrap tuple)
            new_rate = np.random.uniform(*process_library[op])
            
            subsystem.Ls[op] = new_rate 
            
            # note: this directly modifies variable another classe's object
            # ...perhaps could be changed to instead work via method of that class?
            
            model.build_operators()
            
            print('adding operator ' + op + ' to TLS no. ' + str(model.TLSs.index(subsystem))
                  + ' with rate ' + str(new_rate))
            
        else:
            
            print('no options for adding operator to TLS no. ' + str(model.TLSs.index(subsystem)))
        
        
        return model
        
    
    
    def remove_random_L(self, model = False, process_library = False):
        pass
        