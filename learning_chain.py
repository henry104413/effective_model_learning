#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

#import time
import copy
import numpy as np

import learning_model
import params_handling
import process_handling





# single instance executes a learning chain; own hyperparameters including initial model (inc. D) 



class LearningChain():
    
    
    
    
    
    
    
    
    def __init__(self, target_times, target_datasets, target_observables, *,
                 initial = None, # instance of LearningModel or tuple/list of (qubit_energy, defects_number)
                 
                 max_chain_steps = 100,
                 
                 chain_MH_temperature = 0.01,
                 chain_MH_temperature_multiplier = 2,
                 
                 chain_step_options = ['tweak all parameters', 'add L', 'remove L',
                                       'add qubit coupling', 'remove qubit coupling',
                                       'add defect-defect coupling', 'remove defect-defect coupling'],
                 chain_step_probabilities = [10, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02],
                 
                 acceptance_window = 100,
                 acceptance_target = 0.4,
                 acceptance_band = 0.2,
                 
                 params_handler_hyperparams = {
                     'initial_jump_lengths': {'couplings' : 0.001,
                                              'energy' : 0.01,
                                              'Ls' : 0.00001
                                              },
                     },
                 
                 Ls_library = { # will draw from uniform distribution from specified range)
                     'sigmax': (0.001, 0.2)
                    ,'sigmay': (0.001, 0.2)
                    ,'sigmaz': (0.001, 0.2)
                    },
      
                 qubit_couplings_library = { # will draw from uniform distribution from specified range)
                     'sigmax': (-0.05, 0.05)
                    ,'sigmay': (-0.05, 0.05)
                    ,'sigmaz': (-0.05, 0.05)
                    },
                 
                 defect_couplings_library = { # will draw from uniform distribution from specified range)
                     'sigmax': (-0.05, 0.05)
                    ,'sigmay': (-0.05, 0.05)
                    ,'sigmaz': (-0.05, 0.05)
                    }

                 
                 ):
        
        
        
        
        # holders for different model instances in play:
        self.best = None
        self.current = None
        
        # trackers: defined in methods that use them - for now
        self.acceptance_log = []
        
        
        # chain parameters:
        self.max_chain_steps = max_chain_steps
        self.chain_MH_temperature = chain_MH_temperature
        self.chain_MH_temperature_multiplier = chain_MH_temperature_multiplier
        self.chain_step_options = chain_step_options
        self.chain_step_probabilities = chain_step_probabilities
        
        self.acceptance_window = acceptance_window
        self.acceptance_target = acceptance_target
        self.acceptance_band = acceptance_band
        
        
        # default step options (all implemented possibilities) and default probabilities (here equal):
        self.default_step_options = ['tweak all parameters', 'add L', 'remove L',
                                     'add qubit coupling', 'remove qubit coupling',
                                     'add defect-defect coupling', 'remove defect-defect coupling']
        self.default_step_probabilities = [1/len(self.default_step_options) for x in self.default_step_options]
        
        
        # optimiser object and initial hyperparameters for it:
        # (PramsOptimiser instance with attributes for hyperparams
        # ...and methods to carry out params optimisation, set and output hyperparams)
        self.params_handler = None 
        self.initial_params_handler_hyperparams = params_handler_hyperparams
        
        # process handler object:
        # (ModelModifier instance)
        self.process_handler = None
        self.Ls_library = Ls_library 
        self.qubit_couplings_library = qubit_couplings_library
        self.defect_couplings_library = defect_couplings_library
        
        # initial guess model:
        if type(initial) == learning_model.LearningModel:
            self.initial = initial
        elif (type(initial) == tuple or type(initial) == list) and len(initial) == 2:
            self.initial = self.make_initial_model(initial[0], initial[1]) # assuming argument (qubit_energy, defects_number)
        elif type(initial) == int:
            self.initial = self.make_initial_model(1, initial) # assuming initial qubit energy = 1 and argument is defect number
        else:
            raise RuntimeWarning('initial model must be specified:\neither as instance of LearningModel,'
                                 + '\nor tuple or list of (qubit energy, number of defects),'
                                 + '\n or integer number of defects, assuming qubit energy = 1')
            
        # target data:    
        self.target_datasets = target_datasets
        self.target_observables = target_observables
        self.target_times = target_times
        
        
    
    
    def learn(self, chain_step_options = False, chain_step_probabilities = False):
        
        # learning containers:
        self.explored_models = [] # repository of explored models - currently not saved
        self.explored_costs = []
        self.current = copy.deepcopy(self.initial)
        self.best = copy.deepcopy(self.initial)
        self.current_cost = self.cost(self.current)
        self.best_cost = self.current_cost
        
        # acceptance tracking:
        k = 0 # auxiliary iteration counter    
        self.acceptance_tracker = [] # all accept/reject events (bool)
        self.acceptance_ratios_log = [] # acceptance ratios for subsequent windows
        
        
        # step options settings checks:
        
        # if options not passed, use ones set for chain:
        if (not chain_step_options or not chain_step_probabilities): 
            chain_step_options = self.chain_step_options
            chain_step_probabilities = self.chain_step_probabilities
        
        # if options neither passed nor set for chain, do all available steps with default probability:
        if (not chain_step_options or not chain_step_probabilities):
            chain_step_options = self.default_step_options
            chain_step_probabilities = self.default_step_probabilities
            
        # check all step options recognised (assuming default ones all implemented):
        if not set(chain_step_options).issubset(set(self.default_step_options)):
            raise RuntimeError('Some chain step options not recognised')
            
        # check option probabilities array length matches options and normalise if not normalised:
        if len(chain_step_probabilities) != len(chain_step_options):
            raise RuntimeError('array of probabilities of chain step options is the wrong length')
        if (temp := sum(chain_step_probabilities)) != 1:
            chain_step_probabilities = [x/temp for x in chain_step_probabilities]
            
        
    
        # carry out all chain steps:
        for i in range(self.max_chain_steps):
            
            # acceptance tally:
            if k >= self.acceptance_window:
                k = 0
                window_accepted_total = \
                    sum(self.acceptance_tracker[len(self.acceptance_tracker)-self.acceptance_window:len(self.acceptance_tracker)])
                acceptance_ratio = window_accepted_total/self.acceptance_window
                self.acceptance_ratios_log.append(acceptance_ratio)
                
                # adaptation:
                # note: assuming acceptance band is positive = maximum difference either way of ratio and target before adaptation
                if acceptance_ratio - self.acceptance_target > self.acceptance_band: # ie. accepting too much -> cool down
                    self.chain_MH_temperature *= (1/self.chain_MH_temperature_multiplier)
                    #print('cooling down')
                elif acceptance_ratio - self.acceptance_target < -self.acceptance_band: # ie. accepting too little -> heat up
                    self.chain_MH_temperature *= self.chain_MH_temperature_multiplier
                    #print('heating up')
                # !!! add some adaptation method here!!!
                
            k += 1                        
    
            # new proposal container:
            proposal = copy.deepcopy(self.current)
            
            # choose next step and modify proposal accordingly:
            next_step = np.random.choice(chain_step_options, p = chain_step_probabilities)
            if next_step == 'tweak all parameters':
                self.tweak_params(proposal)
                #print('tweaking')
            elif next_step == 'add L':
                self.add_random_L(proposal)
                #print('adding L')
            elif next_step == 'remove L':
                self.remove_random_L(proposal)
                #print('removing L')
            elif next_step == 'add qubit coupling':
                self.add_random_qubit_coupling(proposal)
            elif next_step == 'remove qubit coupling':
                self.remove_random_qubit_coupling(proposal)
            elif next_step == 'add defect-defect coupling':
                self.add_random_defect2defect_coupling(proposal)
            elif next_step == 'remove defect-defect coupling':
                self.remove_random_defect2defect_coupling(proposal)
            
            # evaluate new proposal:
            proposal_cost = self.cost(proposal)
            self.explored_costs.append(proposal_cost)
            
            # if improvement:
            if proposal_cost <= self.current_cost:
                accept = True
            
            # if detriment:
            else: 
                
                # 
                
                # MH criterion:
                # note: also covers improvement for non-zero temperature and this likelihood form
                MH_likelihood = np.exp(-(proposal_cost - self.current_cost)/self.chain_MH_temperature)
                roll = np.random.uniform()
                if roll < MH_likelihood:
                        accept = True
                        
                # rejection otherwise:
                else:
                    accept = False                

            # acceptance:
            if accept:
                
                # update current:
                self.current = proposal
                self.current_cost = proposal_cost
                
                # update best if warranted:
                if proposal_cost < self.best_cost:
                    self.best_cost = proposal_cost
                    self.best = copy.deepcopy(proposal)
                    
                self.acceptance_tracker.append(True)
            
            # rejection:
            else:
                #print('reject')
                self.acceptance_tracker.append(False)
                pass
            
        return self.best
    
    
    
    
    def make_initial_model(self, qubit_energy, defects_number):
    
        """
        Returns empty model with qubit of specified energy,
        given number of defects initialised to qubit energy,
        and no couplings or Lindblad processes.
        """    
    
        initial_model = learning_model.LearningModel()
        initial_model.add_TLS(TLS_label = 'qubit',
                             is_qubit = True,
                             energy = qubit_energy,
                             couplings = {},
                             Ls = {}
                             )
        for i in range(defects_number):
            initial_model.add_TLS(is_qubit = False,
                                  energy = qubit_energy,
                                  couplings = {},
                                  Ls = {}
                                  )
        initial_model.build_operators()
        
        return initial_model
    
    
    

    # calculates and returns cost of model,
    # using target_times and target_data set at instance level,
    # currently using mean squared error between dynamics:
    # note: here is where any weighting or similar should be implemented
    # assumed: either target data is listof numpy arrays and target observables is list of operator labels
    # ...or target data is just numpy array and target observables is single label (in which case instance variables listified)
    
    def cost(self, model):
        
        if isinstance(self.target_datasets, np.ndarray) and isinstance(self.target_observables, str):
            self.target_datasets =  [self.target_datasets]
            self.target_observables = [self.target_observables]
        
        model_datasets = model.calculate_dynamics(evaluation_times = self.target_times, observable_ops = self.target_observables)
        
        # # skip now and if done, redo for all elements of lists... check these are numpy arrays to use array operators below:
   
        # if type(model_data) != type(np.array([])) or type(self.target_data) != type(np.array([])):
   
        #     raise RuntimeError('error calculating cost: arguments must be numpy arrays!\n')
        
        # add up mean-squared-error over different observables, assuming equal weighting:
        # note: now datasets should all be lists of numpy arrays
        total_MSE = 0
        for i in range(len(model_datasets)):
            total_MSE += np.sum(np.square(abs(model_datasets[i]-self.target_datasets[i])))/len(self.target_times)
       
        return total_MSE
    
    
    
     
    
    
    # performs full paramter optimisation on argument model, setting hyperparameterd to chain attribute,
    # saving resulting model to current working model and saving full cost progression and best cost achieved 
    # returns best cost achieved
    
    def optimise_params(self, model_to_optimise):
        
        # note: trackers initialised here as only used by this method, checker avoids resetting on repeat calls
        if not hasattr(self, 'costs_full') and not hasattr(self, 'costs_brief'):
            self.costs_full = [] # parameters optimiser full progression
            self.costs_brief = [] # parameters optimiser only best from call
        
        if not self.params_handler: # ie. first run
            self.params_handler = params_handling.ParamsHandler(self)
        
        self.params_handler.set_hyperparams(self.initial_params_handler_hyperparams)
        self.current, best_cost, costs = self.params_handler.do_optimisation(model_to_optimise)
        self.costs_brief.append(best_cost) 
        self.costs_full = self.costs_full + costs 
        
        return best_cost
        
        
    
    
    # performs a single step in the parameter landscape:
    # works on (ie modifies) argument model, also returns it
    
    def tweak_params(self, model_to_tweak):
        
        # initialise parameters handler if not yet done and set to default hyperparameters
        # note: most hyperparameters only relevant to full optimisation
        if not self.params_handler: # ie. first run
            self.params_handler = params_handling.ParamsHandler(self)
            self.params_handler.set_hyperparams(self.initial_params_handler_hyperparams)
        
        return self.params_handler.tweak_all_parameters(model_to_tweak)
        
    
    
    
    # returns JSON compatible dictionary of hyperparameters (relevant heuristics):
    # namely: initial jump lengths, annealing rate
    
    def chain_hyperparams_dict(self):
       
        chain_hyperparams_dict = {
                                  'initial guess': self.initial.model_description_dict()          
                                  }
        if self.params_handler:
            chain_hyperparams_dict['params optimisation initial hyperparameters'] = self.params_handler.output_hyperparams_init()
        return chain_hyperparams_dict 
    
    
    
    # performs addition of random Linblad process to random subsystem:
    # works on (ie modifies) argument model, also returns it
        
    def add_random_L(self, model_to_modify, Ls_library = False):
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        # unless specified in call, use process library set for chain:
        if not Ls_library:
            Ls_library = self.Ls_library
            
        self.process_handler.add_random_L(model_to_modify)
    
    
    
    # performs removal of random Lindblad process from random subsystem:
    # works on (ie modifies) argument model, also returns it
        
    def remove_random_L(self, model_to_modify):
        
        # ensure process handler exists (created at first run):
        if not self.process_handler: # ie. first run
            self.initialise_process_handler()
            
        self.process_handler.remove_random_L(model_to_modify)
    
    
    
    # performs addition of random symmetric single-operator coupling between random defect and qubit:
    # works on (ie modifies) argument model, also returns it
        
    def add_random_qubit_coupling(self, model_to_modify, qubit_couplings_library = False):
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        # unless specified in call, use process library set for chain:
        if not qubit_couplings_library:
            qubit_couplings_library = self.qubit_couplings_library
            
        self.process_handler.add_random_qubit_coupling(model_to_modify)
        
        
    
    # performs addition of random symmetric single-operator coupling between twp random defects:
    # works on (ie modifies) argument model, also returns it
        
    def add_random_defect2defect_coupling(self, model_to_modify, defect_couplings_library = False):
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        # unless specified in call, use process library set for chain:
        if not defect_couplings_library:
            defect_couplings_library = self.defect_couplings_library
            
    #    self.process_handler.add_random_qubit_coupling(model_to_modify)
        self.process_handler.add_random_defect2defect_coupling(model_to_modify)
    
    
    
    # constructs process handler and sets all process libraries:
        
    def initialise_process_handler(self):
        
        self.process_handler = process_handling.ProcessHandler(self,
                                              qubit_couplings_library = self.qubit_couplings_library,
                                              defect_couplings_library = self.defect_couplings_library,
                                              Ls_library = self.Ls_library)
        
        
    
    def remove_random_qubit_coupling(self, model_to_modify):
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        self.process_handler.remove_random_qubit_coupling(model_to_modify)
    

    def remove_random_defect2defect_coupling(self, model_to_modify):
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        self.process_handler.remove_random_defect2defect_coupling(model_to_modify)
        
        
        
    # !!!ADD METHOD TO UPDATE LIBRARIES OF CURRENT PROCESS HANDLER!!!
                                      