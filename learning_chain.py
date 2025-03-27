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
        # note: only makes sense if product state initially 
        
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
        
        temperature_proposal = 0.00001 # or (0.05, 0.05) to sample gamma by default
        
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
        
        custom_function_on_dynamics_return = False # optional function acting on model's dynamics calculation return
    
    
    
    def complementary_step(self, step_type: str) -> str:
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
                 qubit_initial_state: Qobj = False,
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
                 
                 Ls_library: dict[str, list | tuple] = False,
      
                 qubit2defect_couplings_library: dict[str, list | tuple] = False,
                 
                 defect2defect_couplings_library: dict[str, list | tuple] = False,
                 
                 custom_function_on_dynamics_return: callable = False
                 # optional function acting on model's dynamics calculation return
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
        self.explored_loss = []
        self.current = copy.deepcopy(self.initial)
        self.best = copy.deepcopy(self.initial)
        self.current_loss = self.total_dev(self.current)
        self.best_loss = self.current_loss
        self.chain_windows_acceptance_log = []
        
    
    
    def run(self, steps:int = False) -> learning_model.LearningModel:
        
        """
        Carries out the chain using instance-level hyperparameters;
        populates instance-level progression trackers, returns best model found.
        
        Performs "steps" new proposals if specified, else instance-level value used. 
        """
        
        if not steps: steps = self.max_chain_steps 
        
        # acceptance tracking for this run:
        k = 0 # auxiliary iteration counter    
        self.run_acceptance_tracker = [] # all accept/reject events (bool)
        
        
        # carry out all chain steps:
        for i in range(steps):
            
            # set Metropolis-Hastings acceptance temperature:
            self.MH_temperature = self.sample_T()
            
            # acceptance tally:
            if k >= self.acceptance_window: # ie, end of latest window reached
                print(i) # optional "progress bar"
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
    
            # new proposal container:
            proposal = copy.deepcopy(self.current)
            
            # choose next step:
            next_step = np.random.choice(self.next_step_labels, p = self.next_step_probabilities_list)
            next_step = str(next_step)
            
            # modify proposal accordingly and save number of possible modifications of chosen type:
            proposal, possible_modifications_chosen_type = self.step(proposal, next_step, update = True)
            
            # if no modifications of chosen type possible, skip straight to next proposal iteration:
            # note: this uses up an iteration with no real new proposal and no tracker record
            if not bool(possible_modifications_chosen_type): continue
            
            # also save number of possible modifications of reverse type after performing chosen step:
            # note: proposal not modified by this
            proposal, possible_modifications_reverse_type = self.step(proposal, self.complementary_step(next_step), update = False)
            
            # overall probabilities of making this step and of then reversing it:
            p_there = self.next_step_probabilities_dict[next_step]/possible_modifications_chosen_type
            p_back = self.next_step_probabilities_dict[self.complementary_step(next_step)]/possible_modifications_reverse_type
            # then multiply by back/there
            
            # evaluate new proposal (system evolution calculated here):
            proposal_loss = self.total_dev(proposal)
            self.explored_loss.append(proposal_loss)
            
            # Metropolis-Hastings acceptance:
            acceptance_probability = self.acceptance_probability(self.current, proposal, p_there, p_back)
            if np.random.uniform() < acceptance_probability: # ie. accept proposal
                # update current and also best if warranted:
                self.current = proposal
                self.current_loss = proposal_loss
                if proposal_loss < self.best_loss:
                    self.best_loss = proposal_loss
                    self.best = copy.deepcopy(proposal)
                self.run_acceptance_tracker.append(True)
            else: # ie. reject proposal
                self.run_acceptance_tracker.append(False)
                
         
        #self.chain_windows_acceptance_log.extend(self.run_windows_acceptance_log)   
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
                return self.process_handler.add_random_L(model, update = update)
            case 'remove L':
                return self.process_handler.remove_random_L(model, update = update)
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
                   ) -> float:
        """
        Calculates the Metropolis-Hastings acceptance probability, given:
        current as model or (model, loss), proposal as model or (model, loss),
        probability of moving from current to proposal ("there"),
        probability of reversing that move ("back").
        
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
        """
        
        # terms for formula:
        prior = self.prior # evaluates prior probability of argument model
        T = self.MH_temperature
        current, current_loss = self.process_argument_model(current)
        proposal, proposal_loss = self.process_argument_model(proposal)
        
        # formula (f stands for prior):
        #                                 f(prop) * back
        # exp(-1/T*(MSE(prop)-MSE(curr)))*-------------
        #                                 f(curr) * there
        
        return np.exp(-1/T * (proposal_loss-current_loss)) * prior(proposal) / prior(current) * back / there
    


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


    
    def initialise_process_handler(self):
        """
        Constructs process handler and sets all process libraries.
        """    
        
        self.process_handler = process_handling.ProcessHandler(self,
                                              qubit2defect_couplings_library = self.qubit2defect_couplings_library,
                                              defect2defect_couplings_library = self.defect2defect_couplings_library,
                                              Ls_library = self.Ls_library)
        
        
        
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