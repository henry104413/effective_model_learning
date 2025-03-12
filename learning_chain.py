#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LearningChain class file
Effective model learning
@author: Henry (henry104413)
"""


import copy
import numpy as np

import learning_model
import params_handling
import process_handling



# common types:
TYPE_MODEL = type(learning_model.LearningModel)
    
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
        2) update them in existing parameter and process hadnlers 
    
    Also potentially take parameter handler jump lengths out of outer dictionary.
    """
    
    # bundle of default values for single chain hyperparameters:
    class Defaults:
        
        initial = False # instance of LearningModel or tuple/list of (qubit_energy, defects_number)
        
        max_chain_steps = 100
        chain_step_options = {
            'tweak all parameters': 10,
            'add L': 0.1,
            'remove L': 0.1,
            'add qubit-defect coupling': 0.05, 
            'remove qubit-defect coupling': 0.05,
            'add defect-defect coupling': 0.025, 
            'remove defect-defect coupling': 0.025
            }
        
        temperature_proposal_shape = 0.01 # aka k
        temperature_proposal_scale = 0.01 # aka theta
        
        jump_length_rescaling_factor = 2 # for scaling up or down jump lengths of parameter handler
        
        acceptance_window = 100
        acceptance_target = 0.4
        acceptance_band = 0.2
        
        params_handler_hyperparams = {
            'initial_jump_lengths': {'couplings' : 0.001,
                                     'energy' : 0.01,
                                     'Ls' : 0.00001
                                     }
            }
        
        Ls_library = { # sampled from gamma distribution with given (shape, scale)
            'sigmax': (0.1, 0.5)
           ,'sigmay': (0.1, 0.5)
           ,'sigmaz': (0.1, 0.5)
           }

        qubit2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
            (('sigmax', 'sigmax'),): (0.2, 0.5)
           ,(('sigmay', 'sigmay'),): (0.2, 0.5)
           ,(('sigmaz', 'sigmaz'),): (0.2, 0.5)
           }
        
        defect2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
            (('sigmax', 'sigmay'),): (0.2, 0.5)
           ,(('sigmay', 'sigmay'),): (0.2, 0.5)
           ,(('sigmaz', 'sigmay'),): (0.2, 0.5)
           }
    
    
    
    def complementary_step(step_type: str) -> str:
        """
        Returns step type that reverses argument step type.
        Note: Used in automatic calculation of reverse step probability.
        """
        match step_type:
            case 'tweak all parameters':
                return 'tweak all parameters'
            case 'add L': 
                return 'remove L'
            case 'remove L':
                return 'add L'
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
                 # instance of LearningModel or tuple/list of (qubit_energy, defects_number)
                 
                 # note: arguments below should have counterpart in class Defaults:

                 max_chain_steps: int = False,
                 chain_step_options: dict[str, float | int] = False,
                 
                 temperature_proposal_shape: float = False, # aka k
                 temperature_proposal_scale:float = False, # aka theta
                 
                 jump_length_rescaling_factor: float = False, 
                 
                 acceptance_window: float = False,
                 acceptance_target: float = False,
                 acceptance_band: float = False,
                 
                 params_handler_hyperparams: dict[dict] = False,
                 # note: can contain lots of things - class to be simplified
                 
                 Ls_library: dict[str, list | tuple] = False,
      
                 qubit2defect_couplings_library: dict[str, list | tuple] = False,
                 
                 defect2defect_couplings_library: dict[str, list | tuple] = False
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
                           'target_times', 'target_datasets', 'target_observables']:
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
        # note: run() method throws error if option not implemented - not checked here
        temp = 0 # 
        for option in (options := [x for x in self.chain_step_options]):
            if type(self.chain_step_options[option]) not in [int, float]:
                raise RuntimeError('Chain step options need to have relative probabilities\n'
                                   +'specified by a single number')
            else:
                temp += self.chain_step_options[option]
                
        # step labels, normalised dictionary, and list of just probabilities for later use:
        self.next_step_labels = options # next step labels for use in run()
        self.next_step_probabilities_dict = {option: self.chain_step_options[option]/temp 
                                  for option in options}
        self.next_step_probabilities_list = [self.chain_step_options[option]/temp
                                        for option in options]
        
        # process and parameter objects to perform chain steps:
        # note: initialised at first call of methods that use them
        self.params_handler = None 
        self.process_handler = None
        
        # chain outcome containers:
        self.explored_models = [] # repository of explored models - currently not saved
        self.explored_costs = []
        self.current = copy.deepcopy(self.initial)
        self.best = copy.deepcopy(self.initial)
        self.current_cost = self.total_dev(self.current)
        self.best_cost = self.current_cost
        self.acceptance_log = []
        
    
    
    def run(self):
        
        """
        Carries out the chain using instance-level hyperparameters;
        populates instance-level progression trackers, returns best model found.
        """
        
        # ADD temperature sampling
        self.MH_temperature = 0.00001
        
        # acceptance tracking:
        k = 0 # auxiliary iteration counter    
        self.acceptance_tracker = [] # all accept/reject events (bool)
        self.acceptance_ratios_log = [] # acceptance ratios for subsequent windows
        
        
        # carry out all chain steps:
        for i in range(self.max_chain_steps):
            
            # acceptance tally:
            if k >= self.acceptance_window: # ie, end of last window reached
                k = 0
                window_accepted_total = \
                    sum(self.acceptance_tracker[len(self.acceptance_tracker)-self.acceptance_window:len(self.acceptance_tracker)])
                acceptance_ratio = window_accepted_total/self.acceptance_window
                self.acceptance_ratios_log.append(acceptance_ratio)
                
                # adaptation:
                # note: assuming acceptance band is positive = maximum difference either way of ratio and target before adaptation
                if acceptance_ratio - self.acceptance_target > self.acceptance_band: # ie. accepting too much -> cool down
                    self.cool_down()
                elif acceptance_ratio - self.acceptance_target < -self.acceptance_band: # ie. accepting too little -> heat up
                    self.heat_up()
                    
            k += 1                        
    
            # new proposal container:
            proposal = copy.deepcopy(self.current)
            
            # choose next step:
            next_step = np.random.choice(self.next_step_labels, p = self.next_step_probabilities_list)
            
            # modify proposal and save number of possible modifications of chosen type:
            proposal, possible_modifications_chosen_type = self.step(proposal, next_step, update = True)
            
            # also save number of possible modifications of reverse type after performing chosen step:
            # note: proposal not modified by this
            proposal, possible_modifications_reverse_type = self.step(proposal, self.complementary_step(next_step), update = False)
            
            # overall probabilities of making this step and of then reversing it:
            p_there = self.next_step_probabilities_dict[next_step]/possible_modifications_chosen_type
            p_back = self.next_step_probabilities_dict[self.complementary_step(next_step)]/possible_modifications_reverse_type
            # then multiply by back/there
            
            # evaluate new proposal:
            proposal_cost = self.total_dev(proposal)
            self.explored_costs.append(proposal_cost)
            
            # if improvement:
            if proposal_cost <= self.current_cost:
                accept = True
            
            # if detriment:
            else: 
                
                # MH criterion:
                # note: also covers improvement for non-zero temperature and this likelihood form
                MH_likelihood = np.exp(-(proposal_cost - self.current_cost)/self.MH_temperature)
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
                self.acceptance_tracker.append(False)
                
            
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
            case 'add L': 
                return self.process_handler.add_random_L(model, update)
            case 'remove L':
                return self.process_handler.remove_random_L(model, update)
            case 'add qubit-defect coupling':
                return self.process_handler.add_random_qubit2defect_coupling(model, update)
            case 'remove qubit-defect coupling': 
                return self.process_handler.remove_random_qubit2defect_coupling(model, update)
            case 'add defect-defect coupling':
                return self.process_handler.add_random_defect2defect_coupling(model, update)
            case 'remove defect-defect coupling': 
                return self.process_handler.remove_random_defect2defect_coupling(model, update)
            case _:
                raise RuntimeError('Model proposal (chain step) option \'' 
                                   + step_type + '\' not recognised')
    
    
    
    def total_dev(self, model: TYPE_MODEL) -> float:
        """
        Calculates total deviation of , any realistic amount of net charge on the Moon meansargument model from set target data over set observables. 
        
        Returns equal sum over all instance-level target ovservables
        of mean squared error between instance-level target data
        and argument model data evaluated at instance-level target times.
        
        Assumes target data is list of numpy arrays and target observables is list of operator labels.
        Changes these to lists if currently single array and single label.
        """
        
        if isinstance(self.target_datasets, np.ndarray) and isinstance(self.target_observables, str):
            self.target_datasets =  [self.target_datasets]
            self.target_observables = [self.target_observables]
        
        model_datasets = model.calculate_dynamics(evaluation_times = self.target_times, observable_ops = self.target_observables)
          
        # add up mean-squared-error over different observables, assuming equal weighting:
        # note: now datasets should all be lists of numpy arrays
        total_MSE = 0
        for i in range(len(model_datasets)):
            total_MSE += np.sum(np.square(abs(model_datasets[i]-self.target_datasets[i])))/len(self.target_times)
       
        return total_MSE
    
    
    
    def acceptance_probability(self, 
                   current: TYPE_MODEL, 
                   proposal: TYPE_MODEL, 
                   there: float | int,
                   back: float | int,
                   ) -> float:
        """
        Calculates the Metropolis-Hastings acceptance probability, given:
        current model, proposal model,
        probability of moving from current to proposal ("there"),
        probability of reversing that move ("back").
        
        Also  uses instance-level attributes:
        Metropolis-Hastings temperature - MH_temperature: int|float,
        prior model probability - prior: callable[model: TYPE_MODEL],
        model loss wrt to all observables target data - total_dev: callable[model: TYPE_MODEL]
        
        Incorporates the likelihoods of both models as well as their priors,
        and also the Markov chain reversibility correction factor (* back / there).
        """
        
        prior = self.prior
        T = self.MH_temperature
        loss = self.total_dev
            
        # formula (f stands for prior):
        #                                 f(prop) * back
        # exp(-1/T*(MSE(prop)-MSE(curr)))*-------------
        #                                 f(curr) * there
        
        return np.exp(-1/T * (loss(proposal)-loss(current))) * prior(proposal) / prior(current) * back / there
        
    
    
    def prior(self,
              model: TYPE_MODEL
              ) -> float:
        """
        Calculates the prior probability of argument model.
        The form of this is a heuristic.
        """
    
    
    
    def optimise_params(self,
                        model_to_optimise: TYPE_MODEL
                        ) -> float:
        """
        Currently not maintained.
        
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


    
    def initialise_process_handler(self):
        """
        Constructs process handler and sets all process libraries.
        """    
    
        self.process_handler = process_handling.ProcessHandler(self,
                                              qubit2defect_couplings_library = self.qubit2defect_couplings_library,
                                              defect2defect_couplings_library = self.defect2defect_couplings_library,
                                              Ls_library = self.Ls_library)
        
        
       