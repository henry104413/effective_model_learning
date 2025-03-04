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


# single instance executes a learning chain; own hyperparameters including initial model (inc. D): 

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
            'add qubit coupling': 0.05, 
            'remove qubit coupling': 0.05,
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

    
    
    def __init__(self,
                                  
                 target_times: np.ndarray,
                 target_datasets: np.ndarray | list[np.ndarray],
                 target_observables: str | list[str],
                 *,
                 
                 initial: type(learning_model.LearningModel) | tuple | list = False,
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
        self.init_hyperparams = {} 
        
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
            
        # chain step options check and normalisation:
        # note: run() method throws error if option not implemented - not checked here
        temp = 0 # 
        for option in (options := [x for x in self.chain_step_options]):
            if type(self.chain_step_options[option]) not in [int, float]:
                raise RuntimeError('Chain step options need to have relative probabilities\n'
                                   +'specified by a single number')
            else:
                temp += self.chain_step_options[option]
        self.next_step_labels = options # next step labels for use in run()
        self.next_step_probabilities = [self.chain_step_options[option]/temp for option in options]
        # ^corresponding normalised propabilities
        
        
        # process and parameter objects to perform chain steps:
        # note: initialised at first call of methods that use them
        self.params_handler = None 
        self.process_handler = None
        
        # chain outcome containers:
        self.explored_models = [] # repository of explored models - currently not saved
        self.explored_costs = []
        self.current = copy.deepcopy(self.initial)
        self.best = copy.deepcopy(self.initial)
        self.current_cost = self.total_deviation(self.current)
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
            
            # choose next step and modify proposal accordingly:
            next_step = np.random.choice(self.next_step_labels, p = self.next_step_probabilities)
            if next_step == 'tweak all parameters':
                self.tweak_params(proposal)
            elif next_step == 'add L':
                self.add_random_L(proposal)
            elif next_step == 'remove L':
                self.remove_random_L(proposal)
            elif next_step == 'add qubit coupling':
                self.add_random_qubit2defect_coupling(proposal)
            elif next_step == 'remove qubit coupling':
                self.remove_random_qubit2defect_coupling(proposal)
            elif next_step == 'add defect-defect coupling':
                self.add_random_defect2defect_coupling(proposal)
            elif next_step == 'remove defect-defect coupling':
                self.remove_random_defect2defect_coupling(proposal)
            else:
                raise RuntimeError('Chain step option \'' + next_step + '\' is not implemented') 
            
            # evaluate new proposal:
            proposal_cost = self.total_deviation(proposal)
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
                pass
            
        return self.best
    
    
    
    def make_initial_model(self,
                           qubit_energy: int|float,
                           defects_number: int
                           ) -> type(learning_model.LearningModel):
    
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
    
    
    
    def total_deviation(self, model: type(learning_model.LearningModel)) -> float:

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
                               current: type(learning_model.LearningModel), 
                               proposal: type(learning_model.LearningModel), 
                               step_type: str
                               ) -> float:
        
        # maybe sample new temperature?
        
        # another method calculates probability of moving proposal->current
    
        pass
    
    
    
    # def reversibility_factor(self, next_step):
    #     pass
    #     """
    #     Calculates and returns ratio of prior to marginal,
    #     ie. probability of proposal given current divided by
    #     probability of current given prior (= step reversal).
        
    #     Note: Only processes not present can be added and those present removed,
    #     hence this depends on current model as well as possibilities allowed by process library.
    #     """
    
    #    
    
    
    
    def optimise_params(self,
                        model_to_optimise: type(learning_model.LearningModel)
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
        
    
    
    def tweak_params(self, 
                     model_to_tweak: type(learning_model.LearningModel)
                     ) -> None:
    
        """
        Performs a single step in existing process parameters landscape.
        Modifies argument model, also returns it
        """
        
        # initialise parameters handler if not yet done and set to default hyperparameters
        # note: most hyperparameters only relevant to full optimisation
        if not self.params_handler: # ie. first run
            self.initialise_params_handler()
        return self.params_handler.tweak_all_parameters(model_to_tweak)
    
    
    
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
    
    
        
    def add_random_L(self,
                     model_to_modify: type(learning_model.LearningModel), 
                     Ls_library: dict[str, tuple | list] = False
                     ) -> (type(learning_model.LearningModel), int):
    
        """
        Performs addition of random Linblad process to random subsystem.
        Modifies argument model, also returns it as (model, # possible additions).
        """
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        # unless specified in call, use process library set for chain:
        if not Ls_library:
            Ls_library = self.Ls_library
            
        return self.process_handler.add_random_L(model_to_modify)
    
    
    
    def remove_random_L(self, 
                        model_to_modify: type(learning_model.LearningModel)
                        ) -> (type(learning_model.LearningModel), int):
    
        """
        Performs removal of random Lindblad process from random subsystem.
        Modifies argument model, also returns it as (model, # possible additions).
        """    
        
        # ensure process handler exists (created at first run):
        if not self.process_handler: # ie. first run
            self.initialise_process_handler()
            
        return self.process_handler.remove_random_L(model_to_modify)
    
    
        
    def add_random_qubit2defect_coupling(self,
                                         model_to_modify: type(learning_model.LearningModel),
                                         qubit2defect_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False
                                         ) -> (type(learning_model.LearningModel), int):
         
        """
        Performs addition of random symmetric single-operator coupling between random defect and qubit.
        Modifies argument model, also returns it as (model, # possible additions).
        """
    
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        # unless specified in call, use process library set for chain:
        if not qubit2defect_couplings_library:
            qubit2defect_couplings_library = self.qubit2defect_couplings_library
            
        return self.process_handler.add_random_qubit2defect_coupling(model_to_modify)
        
        
    
    def add_random_defect2defect_coupling(self, 
                                          model_to_modify: type(learning_model.LearningModel),
                                          defect2defect_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False
                                          ) -> (type(learning_model.LearningModel), int):
    
        """
        Performs addition of random symmetric single-operator coupling between twp random defects.
        Modifies argument model, also returns it as (model, # possible additions).
        """    
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        # unless specified in call, use process library set for chain:
        if not defect2defect_couplings_library:
            defect2defect_couplings_library = self.defect2defect_couplings_library
            
        return self.process_handler.add_random_defect2defect_coupling(model_to_modify)
    
    
    
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
        
        
    
    def remove_random_qubit2defect_coupling(self, 
                                            model_to_modify: type(learning_model.LearningModel)
                                            ) -> (type(learning_model.LearningModel), int):
        
        """
        Performs removal of random existing coupling between qubit and defect.
        Modifies argument model, also returns it as (model, # possible removals).
        """    
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        return self.process_handler.remove_random_qubit2defect_coupling(model_to_modify)
    
    

    def remove_random_defect2defect_coupling(self, 
                                            model_to_modify: type(learning_model.LearningModel)
                                            ) -> (type(learning_model.LearningModel), int):
        
        """
        Performs removal of random existing coupling between two defects.
        Modifies argument model, also returns it as (model, # possible removals).
        """    
        
        # ensure process handler exists (created at first run):
        if not self.process_handler:
            self.initialise_process_handler()
            
        return self.process_handler.remove_random_defect2defect_coupling(model_to_modify)
        
        

        
""" !!! to do:

likelihood(temperature, total_deviation)

account for probability of moves - reversibility

temperature sampling

"""        
pass                                      