#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


import time
import copy
import numpy as np

from definitions import Constants
import learning_model
import model

testguy = model.Model()
#testguy = learning_model.LearningModel()

#print(isinstance(testguy, learning_model.LearningModel))
#print(isinstance(testguy, model.Model))



#%%

# executes any modification to model parameters
# stores information on parameter jump lengths, ranges etc (i. e. new proposal distributions)
# also contains its own parameter optimiser
# (simultaneous parameter variation with acceptance/rejection up to specified number of steps)

# methods should take model as argument and work on it
# same parameter handling object expected to be used on various models 

# so methods:
# single step without accept/reject (handled centrally in chain)
# and full optimisation (up to max number of steps) including accept/reject

class ParamsHandler():
    


    # note: constructor takes in chain as argument as keeps it as instance variable
    # instance of ParamsHandler always spawned by chain anyway
    # chain holds target data and also cost method
    # so cost in here always called like: self.chail.cost(model)
    
    # hyperparams can be modified at any time as they can chain throughout chain
    # params optimisation is always done on a model, so has to be called with in as the argument
    # original model passed in will not be modified, but a final best copy will be returned


    def __init__(self, chain, hyperparams = False):
        
        
        # configuration completion indicator -- checked before commencing optimisation:
        
        self.config_done = False
        
        
        # set parameter handler configuration; if not provided, initialise to default values:
        
        if hyperparams:
            
            self.set_hyperparams(hyperparams)
        
        else:
            
            self.set_hyperparams([])
            
            
        # reference to "mother" chain:
        
        self.chain = chain
    
    
    
    
    # sets parameter handler configuration according to argument dictionary;
    # if an attribute has no corresponding entry, sets it to default value specified here:
    
    def set_hyperparams(self, hyperparams):
        
        
        
        # default values (keys are also instance attributes to be set):
        
        default_jump_lengths = {'couplings' : 0.001,
                                'energy' : 0.01,
                                'Ls' : 0.00001
                                }   
        
        default_optimisation_config = {'max_optimisation_steps': int(1e3), 
                        'MH_acceptance': False, 
                        'MH_temperature': 0.1, 
                        'initial_jump_lengths': default_jump_lengths, 
                        'jump_annealing_rate': 0,
                        'acceptance_window': 200,
                        'acceptance_target': 0.4
                        }
        
        
        # set corresponding attributes, also save in dictionary for initial values output:
            
        self.hyperparams_init_output = {}
        
        for key in default_optimisation_config:
            
            if key in hyperparams: # ie. attribute provided
                
                setattr(self, key, hyperparams[key])
                
            else: # else assign default
    
                setattr(self, key, default_optimisation_config[key])
                
            self.hyperparams_init_output[key] = getattr(self, key)
                                
        self.jump_lengths = self.initial_jump_lengths
                
        
        # mark done:
        
        self.config_done = True
        
    
    
    
    # returns dictionary of initial hyperparams:
    
    def output_hyperparams_init(self):
        
        return self.hyperparams_init_output
        
    
        
    
    # returns dictionary of current hyperparams:
    # note: uses keys from initial hyperparams dictionary
        
    def output_hyperparams_curr(self):
        
        return {x: getattr(self, x) for x in self.hyperparams_init_output}
    
    
    
    # interface for Learning Model method to change all existing model parameters:
        
    def tweak_all_parameters(self, model):
    
        if not isinstance(model, learning_model.LearningModel):
            
            raise RuntimeError('Model passed as argument needs to be instance of LearningModel\n'
                               + 'to automatically tweak all parameters!')
            
        if not self.config_done:
            
            raise RuntimeError('Parameter handler hyperparameters need to be specified!')
    
        else:
            
            model.change_params(self.jump_lengths)
        
        
    
    
    # returnes model with optimised parameters starting from argument model:
    
    def do_optimisation(self, initial_model):
        
        """
            
            Carries out full parameter optimisation.
            
            Alters all existing parameters simultaneously,
            with number of steps given by the handler object's attribute max_optimisation_steps.
            
            Acceptance and rejection condition specified here also. Initial model left untouched,
            best model found returned after completion.
            
            Arguments:
                
                initial model ... instance of Learning Model
        
            Returns:
                
                best model
                
                cost of best model
                
                list of costs of all models explored
                
                
        """
        
        
        
        # check hyperparameters and jump lengths set:
        
        if not self.config_done:
            
            raise RuntimeError('Parameter handler hyperparameters not specified!')
    
            
    
        # profiling - time tracking variables:
            
        self.profiling_optimise_params_manipulations = 0.
    
        self.profiling_optimise_params_cost_eval = 0.
        
    
    
        # initialise:
    
        current = copy.deepcopy(initial_model)   
    
        current_cost = self.chain.cost(current)
        
        costs = []
        
        costs.append(current_cost)
        
        best_cost = current_cost
        
        best = copy.deepcopy(current)
        
        acceptance_tracker = np.empty(self.acceptance_window, dtype = bool)
        
        j = 0 # acceptance rate checker auxiliary index
        
        k = 0 # ad hoc plotter auxiliary index
        
        
        
        for i in range(self.max_optimisation_steps): # now fixed steps - could add plateau condition
            
            
            # profiling timer:
                
            clock = time.time()
            
            
            
            acceptance_tracker[j] = True
            
            
            
            # make copy of model, propose new parameters and evaluate cost:
            
            proposed = copy.deepcopy(current) 
        
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
                
                acceptance_tracker[j] = True
                
                if proposed_cost < best_cost:
                    
                    best_cost = proposed_cost
                    
                    best = copy.deepcopy(proposed)
                
                
                
            # if detriment, still accept with given likelihood (like in Metropolis-Hastings)
            # AND update current to proposal (now worse than previous current)
            
            elif (self.MH_acceptance): 
                
                switch_print = False
                  
                if self.MH_temperature == 0:
                    
                    MH_likelihood = 0
                    
                else:
                    
                    MH_likelihood = np.exp(-(proposed_cost - current_cost)/self.MH_temperature)
                
                roll = np.random.uniform()
                
                if switch_print:
                
                    print('__________\niteration: '  + str(i) +  
                          '\nMH likelihood: '+  str(MH_likelihood) +
                          '\nMH roll: ' + str(roll) +
                          '\nold: ' + str(current_cost) +  ', new: ' + str(proposed_cost))
                    
                if roll < MH_likelihood:
                    
                    if switch_print: print('ACCEPT')
                    
                    current = proposed 
                    
                    current_cost = proposed_cost
                    
                    acceptance_tracker[j] = True
                    
                    
                
            
            # else reject: (proposal will be discarded and fresh one made from current at start of loop)
            
            else:
                
                acceptance_tracker[j] = False
               
                pass # proposed will be discarded and a fresh one made from current
            
            
            
            # adapt hyperparameters based on acceptance rate once window completed:
            
            if j == self.acceptance_window - 1:
                
                
                j = 0
                
                acceptance_ratio = acceptance_tracker.sum()/self.acceptance_window
                
                
                
                
            
            else:
                
                j += 1
            
            
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
            
            
            
            
        return best, best_cost, costs
    
        
        
        
        
    # rescales all jump lengths by given factors: (also used for annealing)
    
    def rescale_jump_lengths(self, factor):
        
        
        if not self.config_done:
            
            raise RuntimeError('Parameter handler hyperparameters not specified!')
    
        
        for key in self.jump_lengths:
            
            self.jump_lengths[key] = self.jump_lengths[key]*factor
        
        
        
        