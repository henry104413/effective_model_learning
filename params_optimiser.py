#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import numpy as np

from copy import deepcopy

import time



class ParamsOptimiser():
    



    def _init_(self, hyperparams = False):
        
        
        # configuration completion indicator -- checked before commencing optimisation
        
        self.config_done = False
        
        
        # set parameter optimiser configuration; if not provided, initialise to default values
        
        if hyperparams:
            
            self.set_hyperparams(hyperparams)
        
        else:
            
            self.set_hyperparams([])
    
    
    
    
    # sets parameter optimiser configuration according to argument dictionary;
    # if an attribute has no corresponding entry, sets it to default value specified here:
    
    def set_hyperparams(self, hyperparams):
        
        
        
        
        # default values (keys are also instance attributes to be set):
        
        default_jump_lengths = {}   
        
        default_vals = {'max_steps': int(1e3), 
                        'MH_acceptance': False, 
                        'MH_temperature': 0.1, 
                        'initial_jump_lengths': default_jump_lengths, 
                        'jump_annealing_rate': 0
                        }
        
        
        # set all attributes:
        
        for key in default_vals:
            
            if key in hyperparams: # ie. attribute provided
                
                setattr(self, key, hyperparams[key])
                
            else:
    
                setattr(self, key, default_vals[key])
                
        
        # mark done:
        
        self.config_done = True



    
    # params to define: jump_lengths, starting_model,
    # for optimiser: MH_acceptance, MH_temperature, optimise_params_max_iter (rename to max_steps), jump_annealing_rate
    # need methods: cost, rescale_jump_lengths
    # remove two profiling variables - or keep
    # so could actually passed the chain in and then access the methods
    # would be nice to keep cost there
    # jump lengths method should go here
    
    def optimise_params(self, initial_model):
        
        
        if not self.config_done:
            
            raise RuntimeError('Parameter optimiser hyperparameters not specified!')
    
        # initialise:
    
        if not model_to_optimise: self.current = self.initial
       
        else: self.current = model_to_optimise
        
        current_cost = self.cost(self.current)
        
        costs = []
        
        costs.append(current_cost)
        
        best_cost = current_cost
        
        # self.rescale_jump_lengths(0.1)
        
        counter_MH_accepted = 0
        
        
        
        for i in range(self.max_steps): # add plateau condition
            
            
            # profiling timer:
                
            clock = time.time()
            
            
            # make copy of model, propose new parameters and evaluate cost:
            
            self.proposed = deepcopy(self.current) 
        
            self.proposed.change_params(passed_jump_lengths = self.jump_lengths)
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
                
            clock = time.time()
            
            proposed_cost = self.cost(self.proposed)
            
            self.profiling_optimise_params_cost_eval += (time.time() - clock)
            
            clock = time.time()
            
            costs.append(proposed_cost)
            
            
            
            # anneal jump length:
                
            if self.jump_annealing_rate:
                
                self.rescale_jump_lengths(np.exp(-self.jump_annealing_rate))
            
            
            
            # if improvement -- accept and update current, check if best and save then:
            
            if proposed_cost < current_cost: 
                
                self.current = self.proposed 
                
                current_cost = proposed_cost
                
                if proposed_cost < best_cost:
                    
                    best_cost = proposed_cost
                    
                    self.best = deepcopy(self.proposed)
                
                
                # actually maybe assume that best is always current UNLESS doing MH acceptance!!!!
                
                #ALSO NOPE!!!!
                
            # if detriment, still accept with given likelyhood (like in Metropolis-Hastings)
            # AND update best to current AND update current to proposal (now worse than previous current)
            
            elif (self.MH_acceptance 
                  and np.exp(proposed_cost - current_cost)*self.MH_temperature > np.random.uniform()): 
                
                self.current = self.proposed 
                
                current_cost = proposed_cost
                
                counter_MH_accepted += 1
                
            
            # else reject: (proposal will be discarded and fresh one made from current at start of loop)
            else:
               
                pass # proposed will be discarded and a fresh one made from current
                
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
                
        
        # print('MH acceptance counts: ' + str(counter_MH_accepted))
        
        return costs
    
        # to do: maybe the best model should also be specific to the run of optimise_params (could do more at once)
     
        
test = ParamsOptimiser()

test.set_hyperparams({'MH_acceptance' : True,
                                   'initial_jump_lengths' : {'ok' : 'baby'}
                                 })

