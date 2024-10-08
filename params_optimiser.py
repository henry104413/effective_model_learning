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
    


    # note: constructor takes in chain as argument as keeps it as instance variable
    # instance of ParamsOptimiser always spawned by chain anyway
    # chain holds target data and also cost method
    # so cost in here always called like: self.chail.cost(model)
    
    # hyperparams can be modified at any time as they can chain throughout chain
    # params optimisation is always done on a model, so has to be called with in as the argument
    # original model passed in will not be modified, but a final best copy will be returned


    def _init_(self, chain, hyperparams = False):
        
        
        # configuration completion indicator -- checked before commencing optimisation:
        
        self.config_done = False
        
        
        # set parameter optimiser configuration; if not provided, initialise to default values:
        
        if hyperparams:
            
            self.set_hyperparams(hyperparams)
        
        else:
            
            self.set_hyperparams([])
            
            
        # reference to "mother" chain:
        
        self.chain = chain
    
    
    
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
                                
        self.jump_lengths = self.initial_jump_lengths
                
        
        # mark done:
        
        self.config_done = True



    
    # params to define: jump_lengths, starting_model,
    # need methods: cost, rescale_jump_lengths
    # remove two profiling variables - or keep
    # so could actually passed the chain in and then access the methods
    # would be nice to keep cost there
    # jump lengths method should go here
    
    def optimise_params(self, initial_model):
        
        
        # check hyperparameters and jump lengths set:
        
        if not self.config_done:
            
            raise RuntimeError('Parameter optimiser hyperparameters not specified!')
    
            
        # profiling - time tracking variables:
            
        self.profiling_optimise_params_manipulations = 0.0
    
        self.profiling_optimise_params_cost_eval = 0.0 
    
    
        # initialise:
    
    
        current = deepcopy(initial_model)    
    
        current_cost = self.chain.cost(current)
        
        costs = []
        
        costs.append(current_cost)
        
        best_cost = current_cost
        
        best = deepcopy(current)
        
        
        
        for i in range(self.max_steps): # add plateau condition
            
            
            # profiling timer:
                
            clock = time.time()
            
            
            # make copy of model, propose new parameters and evaluate cost:
            
            proposed = deepcopy(current) 
        
            proposed.change_params(passed_jump_lengths = self.jump_lengths)
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
                
            clock = time.time()
            
            proposed_cost = self.chain.cost(proposed)
            
            self.profiling_optimise_params_cost_eval += (time.time() - clock)
            
            clock = time.time()
            
            costs.append(proposed_cost)
            
            
            
            # anneal jump length:
                
            if self.jump_annealing_rate:
                
                self.rescale_jump_lengths(np.exp(-self.jump_annealing_rate))
            
            
            
            # if improvement -- accept and update current, check if best and save then:
            
            if proposed_cost < current_cost: 
                
                current = proposed 
                
                current_cost = proposed_cost
                
                if proposed_cost < best_cost:
                    
                    best_cost = proposed_cost
                    
                    best = deepcopy(proposed)
                
                
                
            # if detriment, still accept with given likelyhood (like in Metropolis-Hastings)
            # AND update current to proposal (now worse than previous current)
            
            elif (self.MH_acceptance 
                  and np.exp(proposed_cost - current_cost)*self.MH_temperature > np.random.uniform()): 
                
                current = proposed 
                
                current_cost = proposed_cost
                
                
            
            # else reject: (proposal will be discarded and fresh one made from current at start of loop)
            else:
               
                pass # proposed will be discarded and a fresh one made from current
                
            
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
                
        
        
        return best, best_cost, costs
    
        # to do: maybe the best model should also be specific to the run of optimise_params (could do more at once)
        
        
        
        
    # rescales all jump lengths by given factors: (also used for annealing)
    
    def rescale_jump_lengths(self, factor):
        
        
        if not self.config_done:
            
            raise RuntimeError('Parameter optimiser hyperparameters not specified!')
    
        
        for key in self.jump_lengths:
            
            self.jump_lengths[key] = self.jump_lengths[key]*factor
        
        
        
        
test = ParamsOptimiser()

test.set_hyperparams({'MH_acceptance' : True,
                                   'initial_jump_lengths' : {'ok' : 'baby'}
                                 })

