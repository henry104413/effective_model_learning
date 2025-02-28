#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

from __future__ import annotations
import copy
import numpy as np
import typing

import learning_model

if typing.TYPE_CHECKING:
    from learning_chain import LearningChain


class ParamsHandler:
    
    """
    Instance provides access to methods handling existing model parameters. 

    Includes tweak method to simultaneously change all existing parameters except qubit energies,
    each by amount from normal distribution around zero,
    with variance given by instance-level jump length for that class of parameters.

    Currently initialised for specific chain whose target data and loss-related methods
    are accessed by the full optimisation method.
    To do: Potentially remove that mandatory dependence and move this inside optimisation method.
    """

    def __init__(self,
                 chain: type(LearningChain),
                 hyperparams: dict = False
                 ) -> None:
        
        # configuration flag for optimisation method:
        self.config_done = False
        
        # set parameter handler configuration; if not provided, initialise to default values:
        if hyperparams:
            self.set_hyperparams(hyperparams)
        else:
            self.set_hyperparams([])
            
        # reference to "mother" chain:
        self.chain = chain
    
    
    
    def set_hyperparams(self, hyperparams: dict) -> None:
        
        """
        Sets parameter handler hyperparameters according to argument dictionary.
        Those without corresponding entry set to defaults defined here.     
    
        Most related to full optimisation method, jump lengths also used in tweak method.
        """
        
        # default values (keys also determine instance attributes to be set):
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
        self.jump_lengths = copy.deepcopy(self.initial_jump_lengths)
        
        # mark done:
        self.config_done = True
        
    
    
    def output_hyperparams_init(self) -> dict:
        
        """
        Returns dictionary of initial hyperparameters.
        """
        
        return self.hyperparams_init_output
        
    
        
    def output_hyperparams_curr(self) -> dict:
        
        """
        Returns dictionary of instance-level current hyperparameters.
        
        Most related to full optimisation method, jump lengths also used in tweak method.
        """
        
        # note: keys taken from initial hyperparams dictionary
        return {x: getattr(self, x) for x in self.hyperparams_init_output}
    
    
    
    def tweak_all_parameters(self, model: type(learning_model.LearningModel)) -> None:
        
        """
        Tweaks all existing parameters of argument model according instance-level jump lengths.
        
        Calls model method which currently adds to each existing model parameter
        a value sampled from normal distribution around zero
        with variance based on parameter-handler jump length for that type of parameter.
        """
    
        if not isinstance(model, learning_model.LearningModel):
            raise RuntimeError('Model passed as argument needs to be instance of LearningModel\n'
                               + 'to automatically tweak all parameters!')
        if not self.config_done:
            raise RuntimeError('Parameter handler hyperparameters need to be specified!')
        else:
            model.change_params(self.jump_lengths)
        
    
    
    def do_optimisation(self,
                        initial_model: type(learning_model.LearningModel)
                        ) -> type(learning_model.LearningModel):
        
        """
        Note: Method currently not used or maintained.
        
        Carries out full parameter optimisation.
        Modifies argument model.
        Returns best model, total its deviation (aka loss or cost),
        list of total deviations of all models explored.
            
        Alters all existing parameters simultaneously,
        with number of steps given by the handler object's attribute max_optimisation_steps.
         
        Acceptance and rejection condition specified here also. Initial model left untouched,
        best model found returned after completion.
        """
        
        # check hyperparameters and jump lengths set:
        if not self.config_done:
            raise RuntimeError('Parameter handler hyperparameters not specified!')
    
        # initialise:
        current = copy.deepcopy(initial_model)   
        current_cost = self.chain.cost(current)
        costs = []
        costs.append(current_cost)
        best_cost = current_cost
        best = copy.deepcopy(current)
        acceptance_tracker = np.empty(self.acceptance_window, dtype = bool)
        j = 0 # acceptance rate checker auxiliary index
        
        for i in range(self.max_optimisation_steps): # now fixed steps - could add plateau condition
            
            acceptance_tracker[j] = True
            
            # make copy of model, propose new parameters and evaluate cost:
            proposed = copy.deepcopy(current) 
            proposed.change_params(passed_jump_lengths = self.jump_lengths)
            proposed_cost = self.chain.cost(proposed)
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
                if self.MH_temperature == 0:
                    MH_likelihood = 0
                else:
                    MH_likelihood = np.exp(-(proposed_cost - current_cost)/self.MH_temperature)
                roll = np.random.uniform()
                if roll < MH_likelihood:
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
            
        return best, best_cost, costs
    
        
    
    def rescale_jump_lengths(self, factor:int | float) -> None:
        
        """
        Rescales instance-level current jump lengths for all parameters by argument factor.
        """
        
        if not self.config_done:
            raise RuntimeError('Parameter handler hyperparameters not specified!')
    
        for key in self.jump_lengths:
            self.jump_lengths[key] = self.jump_lengths[key]*factor
        
        
        
        