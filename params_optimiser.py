#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import numpy as np

from copy import deepcopy

import time

from definitions import Constants



class ParamsOptimiser():
    


    # note: constructor takes in chain as argument as keeps it as instance variable
    # instance of ParamsOptimiser always spawned by chain anyway
    # chain holds target data and also cost method
    # so cost in here always called like: self.chail.cost(model)
    
    # hyperparams can be modified at any time as they can chain throughout chain
    # params optimisation is always done on a model, so has to be called with in as the argument
    # original model passed in will not be modified, but a final best copy will be returned


    def __init__(self, chain, hyperparams = False):
        
        
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
        
        default_jump_lengths = {'couplings' : 0.001,
                                'energy' : 0.01,
                                'Ls' : 0.00001
                                }   
        
        default_vals = {'max_steps': int(1e3), 
                        'MH_acceptance': False, 
                        'MH_temperature': 0.1, 
                        'initial_jump_lengths': default_jump_lengths, 
                        'jump_annealing_rate': 0,
                        'acceptance_window': 200,
                        'acceptance_target': 0.4
                        }
        
        
        # set corresponding attributes, also save in dictionary for initial values output:
            
        self.hyperparams_init_output = {}
        
        for key in default_vals:
            
            if key in hyperparams: # ie. attribute provided
                
                setattr(self, key, hyperparams[key])
                
            else: # else assign default
    
                setattr(self, key, default_vals[key])
                
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
        
        
    
    
    # returnes model with optimised parameters starting from argument model:
    
    def do_optimisation(self, initial_model):
        
        
        
        # check hyperparameters and jump lengths set:
        
        if not self.config_done:
            
            raise RuntimeError('Parameter optimiser hyperparameters not specified!')
    
            
    
        # profiling - time tracking variables:
            
        self.profiling_optimise_params_manipulations = 0.
    
        self.profiling_optimise_params_cost_eval = 0.
        
    
    
        # initialise:
    
        current = deepcopy(initial_model)   
    
        current_cost = self.chain.cost(current)
        
        costs = []
        
        costs.append(current_cost)
        
        best_cost = current_cost
        
        best = deepcopy(current)
        
        acceptance_tracker = np.empty(self.acceptance_window, dtype = bool)
        
        j = 0 # acceptance rate checker auxiliary index
        
        k = 0 # ad hoc plotter auxiliary index
        
        
        
        for i in range(self.max_steps): # add plateau condition
            
            
            # profiling timer:
                
            clock = time.time()
            
            
            
            acceptance_tracker[j] = True
            
            
            
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
                
                acceptance_tracker[j] = True
                
                if proposed_cost < best_cost:
                    
                    best_cost = proposed_cost
                    
                    best = deepcopy(proposed)
                
                
                
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
                
                
                # note: how the below is done should be discussed and modified!!
                
                # rescale each jump length by ration/target:
                # note: means if accepting too many, take bigger steps - I think that can be bad (stuck in wrong neighbourhood)
                
                # self.rescale_jump_lengths(acceptance_ratio/self.acceptance_target)
                
                
                # rescale M-H temperature - do just this now?
                # note: means if accepting too many, reduce temperature
                
                # self.MH_temperature *= (self.acceptance_target/acceptance_ratio)
                
                # print('scaling temperature by: ' + str(self.acceptance_target/acceptance_ratio))
                
                
            
            else:
                
                j += 1
            
            
            
            self.profiling_optimise_params_manipulations += (time.time() - clock)
            
            
            
            # ad hoc plots:
            
            if k == 200: k = 0    
            
            if k == 0:
                
                filename = 'ad_hoc_' + str(i)
                    
                cost = costs
                    
                import matplotlib.pyplot as plt    
                    
                plt.figure()
                plt.plot(cost, 'm-', linewidth = 0.1, markersize = 0.1)
                plt.yscale('log')
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.xlim([0, 1400])
                plt.ylim([1e-3, 1])
                plt.savefig(filename + '_cost.png', dpi = 1000)
                
                dynamics_ts = self.chain.target_times
                dynamics_datasets = [self.chain.target_data, best.calculate_dynamics(self.chain.target_times)]
                dynamics_labels = ['ground truth', 'learned model']
                
                colours = ['r-', 'b--', 'k:', 'g-.']
                
                # ensure label selector doesn't go out of bounds
                def get_label(i):
                    if not dynamics_labels or len(dynamics_labels) < len(dynamics_datasets): return None
                    else: return dynamics_labels[i]
                
                plt.figure()
                plt.plot(dynamics_ts*Constants.t_to_sec*1e15, dynamics_datasets[0], colours[0], label = get_label(0))
                plt.plot(dynamics_ts*Constants.t_to_sec*1e15, dynamics_datasets[1], colours[1], label = get_label(1))
                
                plt.xlabel('time (fs)')
                plt.ylabel('qubit excited population')
                plt.ylim([0,1.1])
                plt.legend()
                plt.savefig(filename + '_comparison.png', dpi = 1000)
                
            k += 1
        
        
        return best, best_cost, costs
    
        
        
        
        
    # rescales all jump lengths by given factors: (also used for annealing)
    
    def rescale_jump_lengths(self, factor):
        
        
        if not self.config_done:
            
            raise RuntimeError('Parameter optimiser hyperparameters not specified!')
    
        
        for key in self.jump_lengths:
            
            self.jump_lengths[key] = self.jump_lengths[key]*factor
        
        
        
        