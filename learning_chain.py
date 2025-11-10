#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LearningChain class file
Effective model learning
@author: Henry (henry104413)
"""

from __future__ import annotations
import typing
import copy
import numpy as np
import scipy as sp
import time

import learning_model
import params_handling
import process_handling

if typing.TYPE_CHECKING:
    from qutip import Qobj


# common types:
TYPE_MODEL = learning_model.LearningModel # !!! if union here then need to replace type() checks with isinstance below! 
    
class LearningChain:
    
    """
    Single instance executes a reversible-jump Monte Carlo Markov chain given some target data.
    
    Hyperparameters currently set at initialisation. These include chain length in steps,
    as well as properties of upcoming step proposals and acceptance.
    Main method to interact with externally is run(), which returns best model found.
    It also populates instance variables such as progression trackers, 
    which can be accessed externally for plotting etc.
    
    !!! TO ADD:
    Outward facing methods to:
        1) modify chain hyperparameters after initialisation.
        2) update them in existing parameter and process handlers 
    
    Also potentially take parameter handler jump lengths out of outer dictionary.
    """
    
    # bundle of default values for single chain hyperparameters:
    class Defaults:
        
        initial = False # instance of LearningModel or tuple/list of (qubit_energy, defects_number)
        qubit_initial_state = False # instance of Qobj for single qubit 
        defect_initial_state = False # instance of Qobj for single qubit
        # note: only makes sense if product state initially 
        
        max_chain_steps = 10000
        chain_step_options = {
            'tweak all parameters': 10,
            'add qubit L': 1,
            'remove qubit L': 1,
            'add defect L': 1,
            'remove defect L': 1,
            'add qubit-defect coupling': 1, 
            'remove qubit-defect coupling': 1,
            'add defect-defect coupling': 1, 
            'remove defect-defect coupling': 1
            }
        
        temperature_proposal = 0.0005 # or (0.05, 0.05) to sample gamma by default
        
        jump_length_rescaling_factor = 1 # for scaling up or down jump lengths of parameter handler
        
        acceptance_window = 10
        acceptance_target = 0.4
        acceptance_band = 0.2
        
        params_handler_hyperparams = {
            'initial_jump_lengths': {'couplings' : 0.1,
                                     'energies' : 0.1,
                                     'Ls' : 0.01
                                     }
            }
        
        qubit_Ls_library = { # sampled from gamma distribution with given (shape, scale)
            'sigmax': (2, 0.03)
           ,'sigmay': (2, 0.03)
           ,'sigmaz': (2, 0.03)
           }

        defect_Ls_library = { # sampled from gamma distribution with given (shape, scale)
            'sigmax': (2, 0.03)
           ,'sigmay': (2, 0.03)
           ,'sigmaz': (2, 0.03)
           }

        qubit2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                                          # (0.8, 1) currently seems to work well
            (('sigmax', 'sigmax'),): (2, 0.3)
           ,(('sigmay', 'sigmay'),): (2, 0.3)
           ,(('sigmaz', 'sigmaz'),): (2, 0.3)
           }
        
        defect2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
            (('sigmax', 'sigmay'),): (2, 0.3)
           ,(('sigmay', 'sigmay'),): (2, 0.3)
           ,(('sigmaz', 'sigmay'),): (2, 0.3)
           }
        
        params_thresholds = { # minimum values for parameters - if below then process dropped
            # !!! does this break reversibility??                
            'Ls':  1e-7,
            'couplings': 1e-6
            }
        
        params_priors = { # (shape, scale) for gamma dristributions each for one parameter class
            'couplings': (1.04, 30),
            'energies': (1.05, 35),   
            'Ls': (1.004, 23)
            },
        
        params_bounds = { # (lower, upper) bounds when using rejection sampling - False means no rejection sampling to take place
            'couplings': (0.1, 10),
            'energies': (0.01, 1),   
            'Ls': (0.01, 1)
            },
        
        custom_function_on_dynamics_return = False # optional function acting on model's dynamics calculation return
        
        iterations_till_progress_update = False # number of iterations before iteration number and time elapsed printed
    
        store_all_proposals = False # switch to keep all proposed models
    
        
    
    def complementary_step(self, step_type: str) -> str:
        """
        Returns step type that reverses argument step type.
        Note: Used in automatic calculation of reverse step probability.
        """
        match step_type:
            case 'tweak all parameters':
                return 'tweak all parameters'
            case 'add qubit L': 
                return 'remove qubit L'
            case 'remove qubit L':
                return 'add qubit L'
            case 'add defect L': 
                return 'remove defect L'
            case 'remove defect L':
                return 'add defect L'
            case 'add qubit-defect coupling':
                return 'remove qubit-defect coupling' 
            case 'remove qubit-defect coupling': 
                return 'add qubit-defect coupling'
            case 'add defect-defect coupling':
                return 'remove defect-defect coupling' 
            case 'remove defect-defect coupling': 
                return 'add defect-defect coupling'
            case _:
                raise RuntimeError('Complementary step type could not be determined'
                + 'due to step type not recognised')
            
       
        
    def __init__(self,
                                  
                 target_times: np.ndarray,
                 target_datasets: np.ndarray | list[np.ndarray],
                 target_observables: str | list[str],
                 *,
                 
                 initial: TYPE_MODEL | tuple | list = False,
                 qubit_initial_state: Qobj = False,
                 defect_initial_state: Qobj = False,
                 # instance of LearningModel or tuple/list of (qubit_energy, defects_number)
                 
                 # note: arguments below should have counterpart in class Defaults:

                 max_chain_steps: int = False,
                 chain_step_options: dict[str, float | int] = False,
                 
                 temperature_proposal: float|int | tuple[float|int] = False, # value or gamma (shape, scale)
                 
                 jump_length_rescaling_factor: float = False, 
                 
                 acceptance_window: float = False,
                 acceptance_target: float = False,
                 acceptance_band: float = False,
                 
                 params_handler_hyperparams: dict[dict] = False,
                 # note: can contain lots of things - class to be simplified
                 
                 qubit_Ls_library: dict[str, list | tuple] = False,
                 
                 defect_Ls_library: dict[str, list | tuple] = False,
      
                 qubit2defect_couplings_library: dict[str, list | tuple] = False,
                 
                 defect2defect_couplings_library: dict[str, list | tuple] = False,
                 
                 params_thresholds: dict[str, float] = False,
                 # minimum values of parameters below which corresponding process is dropped
                 # discontinued - no more need for filtering
                 
                 params_priors: dict[str, float] = False,
                 
                 params_bounds: dict[str, tuple[float]] = False,
                 
                 custom_function_on_dynamics_return: callable = False,
                 # optional function acting on model's dynamics calculation return
                 
                 iterations_till_progress_update: int = False,
                 # number of iterations before iteration number and time elapsed printed
                 
                 store_all_proposals: bool = False,
                 
                 ):
        
        
        # all argument names and values:
        args = copy.deepcopy(locals())
        
        # container for initial hyperparameters:
        self.init_hyperparams = dict() 
        
        # if passed as false or not passed (False by default),
        # set instance variables to defaults defined for class,
        # otherwise set to passed arguments;
        # then save all in initial hyperparameters dictionary
        # except instance reference, initial guess, target data:
        for key, val in args.items():
            if type(val) == bool and not val:
                val = getattr(self.Defaults, key) # new value taken from Defaults
                setattr(self, key, val)
            else:
                setattr(self, key, val)
            if key not in ['self', 'initial',
                           'target_times', 'target_datasets', 'target_observables',
                           'iterations_till_progress_update']:
                self.init_hyperparams[key] = val
        
        # initial model check or creation if specified only by defects number and qubit energy:
        if (temp := type(self.initial)) == learning_model.LearningModel:
            # ie. already a modifiable model
            pass
        elif (temp == tuple or temp == list) and len(self.initial) == 2:
            self.initial = self.make_initial_model(initial[0], initial[1]) # assuming argument (qubit_energy, defects_number)
        elif temp == int:
            self.initial = self.make_initial_model(1, initial) # assuming initial qubit energy = 1 and argument is defect number
        else:
            raise RuntimeWarning('initial model must be specified:\neither as instance of LearningModel,'
                                 + '\nor tuple or list of (qubit energy, number of defects),'
                                 + '\n or integer number of defects, assuming qubit energy = 1')
            
        # chain step options check and sum for normalisation:
        # note: priorities normalisation retained here for legacy reasons, however another normalisation added
        # ...due to algorithm modification no rescale each priority by # possible modifications of that type (aka # ways)
        # note: run() method throws error if option not implemented - not checked here
        temp = 0 # 
        for option in (options := [x for x in self.chain_step_options]):
            if type(self.chain_step_options[option]) not in [int, float]:
                raise RuntimeError('Chain step options need to have relative priorities\n'
                                   +'specified by a single number')
            else:
                temp += self.chain_step_options[option]
                
        # step labels, normalised dictionary, and list of just priorities for later use:
        self.next_step_labels = options # next step labels for use in run()
        self.next_step_priorities_dict = {option: self.chain_step_options[option]/temp 
                                  for option in options}
        self.next_step_priorities_list = [self.chain_step_options[option]/temp
                                        for option in options] # actually unused as of algorithm of modification with #-ways-scaling
        
        # process and parameter objects to perform chain steps:
        # note: initialised at first call of methods that use them
        self.params_handler = None 
        self.process_handler = None
        
        # chain progression containers:
        self.explored_proposals = [] # repository of explored models
        self.explored_loss = []
        self.explored_acceptance_probability = []
        self.current = copy.deepcopy(self.initial)
        self.best = copy.deepcopy(self.current)
        self.chain_windows_acceptance_log = []
        
        # evaluate initial setup:
        # (immediately filtering parameters below instance-level thresholds)
        self.initialise_process_handler()
        self.process_handler.filter_params(self.current, self.params_thresholds)
        self.current_loss = self.total_dev(self.current)
        self.best_loss = self.current_loss
        self.explored_loss.append(self.current_loss)
        if self.store_all_proposals: self.explored_proposals.append(copy.deepcopy(self.initial))
        
        # counters for overall acceptance tracking (separate for reversible-jump type steps and for value tweak)
        self.tot_RJ_steps = 0
        self.acc_RJ_steps = 0
        self.tot_tweak_steps = 0
        self.acc_tweak_steps = 0
    
    
    def run(self, steps:int = False) -> learning_model.LearningModel:
        
        """
        Carries out the chain using instance-level hyperparameters;
        populates instance-level progression trackers, returns best model found.
        
        Performs "steps" new proposals if specified, else instance-level value used.
        
        Note: 
        If run multiple times, uses current model as starting point (not reinitialised).
        Keeps appending to instance level trackers.
        """
        
        if not steps: steps = self.max_chain_steps 
        
        # acceptance tracking for this run:
        k = 0 # auxiliary iteration counter    
        self.run_acceptance_tracker = [] # all accept/reject events (bool)
        
        # progress tracking (also used in redirected output):
        time_last = time.time() # elapsed time (s)
        k2 = 0
        
        # carry out all chain steps:
        for i in range(steps):
            
            # set Metropolis-Hastings acceptance temperature:
            self.MH_temperature = self.sample_T()
            
            # acceptance tally:
            if k >= self.acceptance_window: # ie, end of latest window reached
                k = 0
                window_accepted_total = \
                    sum(self.run_acceptance_tracker[len(self.run_acceptance_tracker)-
                                                    self.acceptance_window : len(self.run_acceptance_tracker)])
                acceptance_ratio = window_accepted_total/self.acceptance_window
                self.chain_windows_acceptance_log.append(acceptance_ratio)
                
                # adaptation:
                # note: assuming acceptance band is positive = maximum difference either way of ratio and target before adaptation
                if acceptance_ratio - self.acceptance_target > self.acceptance_band: # ie. accepting too much -> cool down
                    self.cool_down()
                elif acceptance_ratio - self.acceptance_target < -self.acceptance_band: # ie. accepting too little -> heat up
                    self.heat_up()
            k += 1

            # progress timing:
            if bool(self.iterations_till_progress_update):
                if k2 >= self.iterations_till_progress_update:
                    k2 = 0
                    print('\n\nIteration:\n' + str(i) + '\nElapsed (s):\n' + 
                          str(np.round((new_time := time.time()) - time_last,2)), flush = True)
                    time_last = new_time
                k2 += 1
                
    
            # new proposal container:
            proposal = copy.deepcopy(self.current)
            
            # choose next step:
            # instead of choosing and only then considering how many ways there were to do this and reversal
            # choose based on priority AND how many options there are
            # make and normalise list of probabilities in order of self.next_step_labels
            # note: product priority (normalised earlier but that is irrelevant) and number of possible modifications of step type (# ways)
            # note: 2nd element step() return is # ways
            next_step_probabilities_list = [self.next_step_priorities_dict[x]*self.step(proposal, x, update = False)[1]
                                            for x in self.next_step_labels]
            next_step_probabilities_list = [x/sum(next_step_probabilities_list) for x in next_step_probabilities_list]     
            
            # choose next step:
            next_step = np.random.choice(self.next_step_labels, p = next_step_probabilities_list)
            next_step = str(next_step)
            
            # update total counter for appropriate step type:
            if next_step == 'tweak all parameters':
                self.tot_tweak_steps += 1
            else: # ie. reversible-jump type step
                self.tot_RJ_steps += 1
            
            # modify proposal accordingly and save number of possible modifications of chosen type:
            proposal, possible_modifications_chosen_type = self.step(proposal, next_step, update = True)
            
            
            # overall probabilities of making this step and of afterwards reversing it:
            if next_step == 'tweak all parameters':
                # based on truncated gaussian distribution formulas    
                p_there = 1 
                p_back = 1
                
                if not self.params_bounds:
                # no bounds specified hence only parameter restriction L positivity    
                    proposal_width = self.params_handler.jump_lengths['Ls']
                    for TLS in self.current.TLSs:
                        for current_rate in TLS.Ls.values():
                            p_there *= 1/(1-1/2*(1+sp.special.erf(-current_rate/proposal_width/np.sqrt(2))))
                    for TLS in proposal.TLSs:
                        for proposed_rate in TLS.Ls.values():
                            p_back *= 1/(1-1/2*(1+sp.special.erf(-proposed_rate/proposal_width/np.sqrt(2))))
                
                else:
                # bounds specified hence rejection sampling applies to all parameters;
                # go over all existing parameters in self.current and in proposal to build there and back terms
                # note: bounds and jump width set for each parameter class
                
                    # note: have to use something mutable in zip loop below!!! 
                    # e. g.: for key, n in zip(holder, Ns): holder[key] *= 2,
                    # with holder = {'key1': 1, 'key2': 5}
                    # multiply by: xi(parameter, width, low bound, high bound)
                    # i. e.: holder[key] *= self.xi(m = , s = , a = , b = )
                    holder = {'p_there': p_there, 'p_back': p_back}
                    for model, key in zip([proposal, self.current], holder):
                        # build products in holder[key] using all parameters in model 
                        # note:
                        # back = product of xi(current)
                        # there = product of xi(proposal)
                        for TLS in model.TLSs:
                            # energy:
                            m = TLS.energy
                            s = self.params_handler.jump_lengths['energies']
                            a = self.params_bounds['energies'][0]
                            b = self.params_bounds['energies'][1]
                            holder[key] *= self.xi(m, s, a, b)
                            # note: this should cancel out if qubit as currently not tweaking their energies
                            
                            # Ls:
                            s = self.params_handler.jump_lengths['Ls']
                            a = self.params_bounds['Ls'][0]
                            b = self.params_bounds['Ls'][1]
                            for m in TLS.Ls.values():
                                holder[key] *= self.xi(m, s, a, b)
                                
                            # couplings:
                            s = self.params_handler.jump_lengths['couplings']
                            a = self.params_bounds['couplings'][0]
                            b = self.params_bounds['couplings'][1]
                            for partner in TLS.couplings:
                                for coupling in TLS.couplings[partner]: # coupling is a touple (strength, [(op1, op2),...])
                                    m = coupling[0]
                                    holder[key] *= self.xi(m, s, a, b)
                                          
                    p_there = holder['p_there']
                    p_back = holder['p_back']
                                     
                    
            else: # ie. a process addition or removal
                # priority of this normalised by sum off all priorities times their respective # ways
                # in numerator # of ways here cancels with that in step choice probability, leaving only priority
                p_there = (self.next_step_priorities_dict[next_step]
                           / sum([self.next_step_priorities_dict[x] * self.step(self.current, x, update = False)[1]
                                  for x in self.next_step_labels]))
                p_back  = (self.next_step_priorities_dict[self.complementary_step(next_step)]
                           / sum([self.next_step_priorities_dict[x] * self.step(proposal, x, update = False)[1]
                                  for x in self.next_step_labels]))
                
                
            # calculate priors ratio due to all paramteres of proposal and current if applying tweak step:
            # note: not present when adding or removing processes (reversible jumps)
            if next_step == 'tweak all parameters':
                params_priors_ratio = 1
                # go over existing parameters in proposal and current and multiply and divide respectively
                # by probability density function given by each prior (currently based on process class)
                
                if not self.params_bounds:    
                # using gamma priors if no bounds specified:
                    
                    # current:
                    for TLS in self.current.TLSs:
                        # energy:
                        x = TLS.energy
                        shape, scale = self.params_priors['energies']
                        params_priors_ratio /= sp.stats.gamma.pdf(x, a=shape, scale=scale)
                        
                        # Ls:
                        shape, scale = self.params_priors['Ls']
                        for x in TLS.Ls.values():
                            params_priors_ratio /= sp.stats.gamma.pdf(x, a=shape, scale=scale)
                            
                        # couplings:
                        shape, scale = self.params_priors['couplings']
                        for partner in TLS.couplings:
                            for coupling in TLS.couplings[partner]: # coupling is a touple (strength, [(op1, op2),...])
                                x = coupling[0]
                                params_priors_ratio /= sp.stats.gamma.pdf(x, a=shape, scale=scale)
                                
                    # proposal:
                    for TLS in proposal.TLSs:
                        # energy:
                        x = TLS.energy
                        shape, scale = self.params_priors['energies']
                        params_priors_ratio *= sp.stats.gamma.pdf(x, a=shape, scale=scale)
                        
                        # Ls:
                        shape, scale = self.params_priors['Ls']
                        for x in TLS.Ls.values():
                            params_priors_ratio *= sp.stats.gamma.pdf(x, a=shape, scale=scale)
                            
                        # couplings:
                        shape, scale = self.params_priors['couplings']
                        for partner in TLS.couplings:
                            for coupling in TLS.couplings[partner]: # coupling is a touple (strength, [(op1, op2),...])
                                x = coupling[0]
                                params_priors_ratio *= sp.stats.gamma.pdf(x, a=shape, scale=scale)
                                
                else:   
                # otherwise (ie. bounds specified) asume uniform distribution and everything cancels:
                    params_priors_ratio *= 1
                    
                    
            else: # ie. a process addition or removal
                params_priors_ratio = 1
            
                                  
            # evaluate new proposal (system evolution calculated here):
            proposal_loss = self.total_dev(proposal)
            self.explored_loss.append(proposal_loss)
            proposal.disp()
            
            # Metropolis-Hastings acceptance:
            acceptance_probability = self.acceptance_probability(self.current, proposal, p_there, p_back, 
                                                                 params_priors_ratio)
            self.explored_acceptance_probability.append(acceptance_probability)
            if np.random.uniform() < acceptance_probability: # ie. accept proposal
                # update current and also best if warranted:
                self.current = proposal
                self.current_loss = proposal_loss
                if proposal_loss < self.best_loss:
                    self.best_loss = proposal_loss
                    self.best = copy.deepcopy(proposal)
                self.run_acceptance_tracker.append(True)
                # save accepted proposal for statistical analysis of chain
                if self.store_all_proposals:
                    self.explored_proposals.append(copy.deepcopy(proposal))
                # update accepted step counter:
                if next_step == 'tweak all parameters':
                    self.acc_tweak_steps += 1
                else: # ie. reversible-jump type step
                    self.acc_RJ_steps += 1
                
            else: # ie. reject proposal
                self.run_acceptance_tracker.append(False)
                
         
        if bool(self.iterations_till_progress_update):
            print('\n\nChain run completed.\n'
                  +'_________________________\n\n', flush = True)
        self.best.final_loss = self.best_loss
        
        self.all_proposals = {'proposals': self.explored_proposals,
                              'loss': self.explored_loss,
                              'acceptance': self.run_acceptance_tracker,
                              'acceptance_probability': self.explored_acceptance_probability
                             } 
        
        return self.best
    
    
    
    def make_initial_model(self,
                           qubit_energy: int|float,
                           defects_number: int
                           ) -> TYPE_MODEL:
        """
        Returns empty model with qubit of specified energy,
        given number of defects initialised to qubit energy,
        and no couplings or Lindblad processes.
        """    
    
        initial_model = learning_model.LearningModel()
        initial_model.add_TLS(TLS_label = 'qubit',
                             is_qubit = True,
                             initial_state = self.qubit_initial_state,
                             energy = qubit_energy,
                             couplings = {},
                             Ls = {}
                             )
        for i in range(defects_number):
            initial_model.add_TLS(is_qubit = False,
                                  initial_state = self.defect_initial_state,
                                  energy = 0.1, # !!! AD HOC - maybe move this to configs 
                                  couplings = {},
                                  Ls = {}
                                  )
        initial_model.build_operators()
        
        return initial_model
    
    
    
    def step(self,
             model: TYPE_MODEL, 
             step_type: str, 
             update: bool = True
             ) -> tuple[TYPE_MODEL, int]:
        """
        Calls methods of process and parameter handlers correspoing step type string.
        If update flag on: carries out modification on argument model.
        If update flag off: modification NOT performed but corresponding method called 
        to evaluate possible proposals - used in reversal probability calculation.
        In both cases returns:
        (model, # possible proposals of specified type).
        
        Currently proposal hyperameters are set at chain initialisation and passed to 
        process and parameter handlers initialisers when instantiated for this chain.
        !!! To do: Integrate methods changing hyperparameters during chain run if required.
        """
        
        # ensure process and parameter handlers instantiated:
        if not self.process_handler: self.initialise_process_handler()
        if not self.params_handler: self.initialise_params_handler()
        
        # call handler method corresponding to step type:
        match step_type:
            case 'tweak all parameters':
                # now just returns 1 as # possible proposals
                # this way reversal probability calculation assumes parameter reversal equally likely
                # !!! updating proposal variance throughout may violate this?
                if update: 
                    return (self.params_handler.tweak_all_parameters(model), 1)
                if not update:
                    return (model, 1)
            case 'add qubit L': 
                return self.process_handler.add_random_qubit_L(model, update = update)
            case 'add defect L': 
                return self.process_handler.add_random_defect_L(model, update = update)
            case 'remove qubit L':
                return self.process_handler.remove_random_qubit_L(model, update = update)
            case 'remove defect L':
                return self.process_handler.remove_random_defect_L(model, update = update)
            case 'add qubit-defect coupling':
                return self.process_handler.add_random_qubit2defect_coupling(model, update = update)
            case 'remove qubit-defect coupling': 
                return self.process_handler.remove_random_qubit2defect_coupling(model, update = update)
            case 'add defect-defect coupling':
                return self.process_handler.add_random_defect2defect_coupling(model, update = update)
            case 'remove defect-defect coupling': 
                return self.process_handler.remove_random_defect2defect_coupling(model, update = update)
            case _:
                raise RuntimeError('Model proposal (chain step) option \'' 
                                   + step_type + '\' not recognised')
    
    
    
    def total_dev(self, model: TYPE_MODEL) -> float:
        """
        Calculates total deviation of argument model from target data. 
        
        Returns equal sum over all instance-level target observables
        of mean squared error between instance-level target data
        and argument model data evaluated at instance-level target times.
        
        Assumes target data is list of numpy arrays,
        and target observables is same-length list of corresponding observable labels.
        Also assumes all data arrays and times are same length.
        Encapsulates data and observable in lists if single array and single label.
        """
        
        if isinstance(self.target_datasets, np.ndarray) and isinstance(self.target_observables, str):
            self.target_datasets =  [self.target_datasets]
            self.target_observables = [self.target_observables]
        
        model_datasets = model.calculate_dynamics(evaluation_times = self.target_times, 
                                                  observable_ops = self.target_observables,
                                                  custom_function_on_return = self.custom_function_on_dynamics_return)
          
        # add up mean-squared-error over different observables, assuming equal weighting:
        # note: now datasets should all be lists of numpy arrays
        total_MSE = 0
        for i in range(len(model_datasets)):
            total_MSE += np.sum(np.square(abs(model_datasets[i]-self.target_datasets[i])))/len(self.target_times)
       
        return total_MSE
    
    
    
    def acceptance_probability(self, 
                   current: TYPE_MODEL | tuple[TYPE_MODEL, int | float], 
                   proposal: TYPE_MODEL | tuple[TYPE_MODEL, int | float], 
                   there: float | int,
                   back: float | int,
                   params_priors_ratio: float = 1
                   ) -> float:
        """
        Calculates the Metropolis-Hastings acceptance probability, given:
        current as model or (model, loss), proposal as model or (model, loss),
        probability of moving from current to proposal ("there"),
        probability of reversing that move ("back"),
        prior multiplier factor as a function of parameters - currently taken from run method.
        
        Also  accesses instance-level attributes:
        Metropolis-Hastings temperature - MH_temperature: int|float,
        prior model probability - prior: callable[model: TYPE_MODEL],
        argument model handler - process_argument_model: callable[arg: TYPE_MODEL|tuple[TYPE_MODEL, int | float]]
        
        Incorporates likelihoods of both models as well as their priors,
        and also Markov chain reversibility correction factor (* back / there).
        
        Currently loss is total deviation from chain instance level target data,
        as equal sum over all specified observables.
        Loss calculated here if only model references passed, otherwise passed value used.
        Note: Purpose is to avoid expensive loss recalculation,
        since loss values commonly used and stored outside this method.
        
        Note: in current setup the prior method only depends on model complexity;
        - as step type held locally within run method and parameter prior factor formulation depends on it,
        it is evaluated therein and ratio passed here by value.
        """
        
        # terms for formula:
        prior = self.prior # evaluates prior probability of argument model
        T = self.MH_temperature
        current, current_loss = self.process_argument_model(current)
        proposal, proposal_loss = self.process_argument_model(proposal)
        
        # formula (f stands for prior of model):
        #                                 f(prop) * back
        # exp(-1/T*(MSE(prop)-MSE(curr)))*---------------
        #                                 f(curr) * there
        # note: function calls currently only incorporate complexity,
        # reversible jump has no further prior beyond proposal distribution,
        # parameters-tweak prior ratio is evaluated in run method and passed here as value
        
        return (np.exp(-1/T * (proposal_loss-current_loss)) 
                * prior(proposal) / prior(current) 
                * back / there 
                * params_priors_ratio)
    


    def process_argument_model(self,
                               arg: TYPE_MODEL | tuple[TYPE_MODEL, int | float]
                               ) -> (TYPE_MODEL, float):
        """
        Always returns (model, loss of model).
        Arguments are either just model, or (model, loss of model);
        note: in latter case just returns arguments as is. 
        
        Loss calculation uses instance level total_dev: callable[model: TYPE_MODEL]
        - currently total deviation from chain instance level target data,
        as equal sum over all specified observables.
        
        Note: Purpose is to simplify input handling in other methods,
        usually where they can be called on just models for generality,
        but are often called with loss precalculated to avoid repetition
        of expensive loss calculation.
        """
        
        if isinstance(arg, TYPE_MODEL):
            return (arg, self.total_dev(arg))
        elif (type(arg) == tuple and len(arg) == 2 
              and isinstance(arg[0], TYPE_MODEL) and isinstance(arg[1], int|float)):
            return (arg[0], arg[1]) # ie. return arg as is
        else:
            raise RuntimeError('Learning chain\'s process_argument_model method failed: invalid input')
                  
    
    
    def prior(self,
              model: TYPE_MODEL
              ) -> float:
        """
        Calculates the prior probability of argument model.
        The form of this is a heuristic.
        """
        total_complexity = (self.process_handler.count_Ls(model)
                            + self.process_handler.count_defect2defect_couplings(model)
                            + self.process_handler.count_qubit2defect_couplings(model))
        return np.exp(-1*(total_complexity)) 
    
    
    
    def optimise_params(self,
                        model_to_optimise: TYPE_MODEL
                        ) -> float:
        """
        Currently deprecated.
        
        Performs full paramter optimisation on argument model. 
        
        Uses hyperparameters set for chain instance.
        Saves resulting model to current working model and full cost progression and best cost achieved. 
        Returns best cost achieved.
        """
        
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
        
    
    
    def cool_down(self):
        """
        Scale down parameter handler jump length by instance-level rescaling factor.
        """
        
        if not self.params_handler: # ie. first run
            self.initialise_params_handler()
        self.params_handler.rescale_jump_lengths(1/self.jump_length_rescaling_factor)
        
        
    
    def heat_up(self):
        """
        Scale up parameter handler jump length by instance-level rescaling factor.
        """
        
        if not self.params_handler: # ie. first run
            self.initialise_params_handler()
        self.params_handler.rescale_jump_lengths(self.jump_length_rescaling_factor)
    
    
    
    def get_init_hyperparams(self):
        """
        Returns JSON compatible dictionary of initial chain hyperparameters.    
        """
        
        output = copy.deepcopy(self.init_hyperparams)
        output['initial guess'] = self.initial.model_description_dict()          
        return output
    
    
    
    def initialise_params_handler(self):
        """
        Constructs parameters handler and sets initial hyperparameters (including jump lenghts).
        """    

        self.params_handler = params_handling.ParamsHandler(self)
        self.params_handler.set_hyperparams(self.params_handler_hyperparams)
        self.params_handler.set_bounds(self.params_bounds)


    
    def initialise_process_handler(self):
        """
        Constructs process handler and sets all process libraries.
        """    
        
        self.process_handler = process_handling.ProcessHandler(self,
                                              bounds = self.params_bounds,                 
                                              qubit2defect_couplings_library = self.qubit2defect_couplings_library,
                                              defect2defect_couplings_library = self.defect2defect_couplings_library,
                                              qubit_Ls_library = self.qubit_Ls_library,
                                              defect_Ls_library = self.defect_Ls_library)
        
        
        
    def sample_T(self):
        """
        Returns temperature for Metropolis-Hastings acceptance.
        Does not directly modify instance variable.
        
        Based on instance level temperature_proposal:            
        If number, returns this value.
        If tuple of numbers (shape, scale), returns value sampled from such gamma distribution.
        """
        
        match self.temperature_proposal:
            case int() | float(): return self.temperature_proposal
            case (int()|float(), int()|float()): return np.random.gamma(*self.temperature_proposal)
            case _: raise RuntimeError('Metropolis-Hastings temperature proposal failed')



    def trunc_gaus(self, x, m, s, a, b):
        """
        Returns probability density function at x
        of normal distribution with stdev s and mean m,
        if truncated at bounds a and b, a < b.
        """
        
        # shorthands:
        def psi(y):
            return 1/np.sqrt(2*np.pi)*np.exp(-1/2*y**2)
        def phi(y):
            return 1/2*(1 + sp.special.erf(y/np.sqrt(2)))
        
        return 1/s * (psi((x-m)/s)) / (phi((b-m)/s) - phi((a-m)/s))
    
    
    
    def xi(self, m:float|int,
           s: float|int,
           a: float|int = -np.inf,
           b: float|int = np.inf):
        """
        Returns value of denominator expression in truncated normal distribution probability density.
        In the ratio when assuming stdev s, lower bound a, higher bound b all constant throughout the chain,
        the numerators and the 1/s factors cancel, leaving only dependence on a, b, s,
        as well as mean m (old or current parameter value before making change).
        """
        
        # shorthands:
        def phi(y):
            return 1/2*(1 + sp.special.erf(y/np.sqrt(2)))
        
        return phi((b-m)/s) - phi((a-m)/s)
    
    
    
    