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
    
    
    
    def __init__(self, chain = False, model = False, Ls_library = False):

        self.Ls_library = Ls_library   # dictionary: {'op label': (lower_bound, upper_bound)}
        self.model = model # only for future methods tied to specific model
        self.chain = chain # only for future methods tied to chain (eg using its cost function)

        

    # decides and carries out either addition or removal of a random L on a random defect
    # - currently deprecated
    
    def add_toss_L(self, model = False, Ls_library = False):
        
            
        
        # set model and process library if passed now, then check both set to appropriate types:
        
        if isinstance(model, LearningModel) or isinstance(model, Model):
            
            self.model = model
            
        if isinstance(Ls_library, dict):
            
            self.Ls_library = Ls_library
            
        if not (isinstance(self.Ls_library, dict)):
            
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
        
            options = [x for x in self.Ls_library if x not in existing]
                
            if options:
            
                op = np.random.choice(options)
                
                subsystem.Ls[op] = self.Ls_library[op]
                
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
        
        return None
            
        
            
    # adds random new single-site Linblad process from process library to random subsystem (qubit or defect for now) 
    # with rate drawn from uniform distribution between bounds given by tuple for each process library entry  
    # works on and returns argument model
    # argument process library used if passed, otherwise use that of handler if already set, error if neither available  
            
    def add_random_L(self, model, Ls_library = False):
        
        # check library available:
        if Ls_library == False:
            if isinstance(self.Ls_library, dict):
                Ls_library = self.Ls_library
            else:
                raise RuntimeError('Cannot add Lindblad process due to process library not specified')
            
        # select subsystem:
        subsystems = [x for x in model.TLSs]# if not x.is_qubit]
        subsystem = np.random.choice(subsystems)
        
        # identify Ls in library not yet present on selected subsystem:
        existing = [x for x in subsystem.Ls]
        options = [x for x in Ls_library if x not in existing] 
        
        # select new operator if available, draw rate from uniform distribution between bounds, and add:
        if options:
            operator = np.random.choice(options)
            new_rate = np.random.uniform(*Ls_library[operator])
            subsystem.Ls[operator] = new_rate 
            model.build_operators()
            #print('adding operator ' + operator + ' to TLS no. ' + str(model.TLSs.index(subsystem))
            #     + ' with rate ' + str(new_rate))
            # note: this directly modifies variable another classe's object
            # ...perhaps could be changed to instead work via method of that class?
            
        else:
            #print('no options for adding operator to TLS no. ' + str(model.TLSs.index(subsystem)))
            pass
        
        return model
        
    
    
    # removes random Lindblad process from random subsystem
    # works on and returns argument model
    
    def remove_random_L(self, model = False):
        
        # select subsystem:
        subsystems = [x for x in model.TLSs]# if not x.is_qubit]
        subsystem = np.random.choice(subsystems)
        
        # identify Ls present on selected subsystem:
        options = [x for x in subsystem.Ls]
        
        # select operator and remove:
        if options:
            operator = np.random.choice(options)
            subsystem.Ls.pop(operator)
            #print('removing operator ' + operator + ' from TLS no. ' + str(model.TLSs.index(subsystem)))
            
        else:
            #print('no options for removing operator from TLS no. ' + str(model.TLSs.index(subsystem)))
            pass
        
        
    # sets process library post-constructor : 
    def define_Ls_library(self, Ls_library):
    
        self.Ls_library = Ls_library
        
    
    
    # rescales bound distance from middle of range by half of given factor for all process library rates:
    # note: range hence changes by up to factor
    # rates assumed positive and capped at zero from left (hence range can change less)
    def rescale_Ls_library_range(self, factor):
    
        if type(self.Ls_library) == bool and not self.Ls_library:
            raise RuntimeError('Process library not retained by handler so cannot rescale range')
    
        for operator in self.Ls_library:
            
            # note: bounds are currently tuples and not mutable!
            
            # calculate new bounds:
            old_bounds = self.Ls_library[operator]
            middle = (old_bounds[1] + old_bounds[0])/2
            old_distance = old_bounds[1] - old_bounds[0]
            new_offset = old_distance*factor/2
            new_bounds = (max(0, middle - new_offset), middle + new_offset)
            
            # save in library:
            self.Ls_library[operator] = new_bounds