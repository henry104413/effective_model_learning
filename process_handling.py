#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

from __future__ import annotations
import typing
import copy
import numpy as np

import basic_model
import learning_model
import two_level_system

if typing.TYPE_CHECKING:
    from learning_chain import LearningChain


COUPLING_LIB_TYPE = dict[tuple[tuple[str] | str], tuple[int | float]] 

class ProcessHandler:
    
    """
    Instance provides access to methods adding and removing model processes,
    ie. coupling terms in Hamiltonian or Lindblad operators.
    
    Holds libraries of available processes and corresponding distributions their parameters are sampled from.
    """    
    
    def __init__(self,
                 chain: type(LearningChain) = False,
                 model: type(basic_model) | type(learning_model) = False,
                 Ls_library: dict[str, tuple | list] = False, # {'op label': (shape, scale)}
                 qubit_couplings_library: COUPLING_LIB_TYPE = False,
                 defect_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False
                 # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                 ):

        self.Ls_library = Ls_library
        self.initial_Ls_library = copy.deepcopy(Ls_library)
        self.qubit_couplings_library = qubit_couplings_library
        self.defect_couplings_library = defect_couplings_library
        self.initial_qubit_couplings_library = copy.deepcopy(qubit_couplings_library)
        self.model = model # only for future methods tied to specific model
        self.chain = chain # only for future methods tied to chain (eg using its cost function)

        

    def add_random_L(self,
                     model: type(basic_model.BasicModel) | type(learning_model.LearningModel),
                     Ls_library: dict[str, tuple | list] = False, # {'op label': (shape, scale)}
                     update: bool = True
                     ) -> (type(basic_model.BasicModel) | type(learning_model.LearningModel), int):
        
        """
        Can add random new single-site Linblad process from process library to random subsystem.
        Modifies argument model and returns: updated model, number of addable Ls.
        Argument library used if passed, instance-level one otherwise, error if neither available.  
        Update flag true means addition performed; false avoids changing model,
        to only get count of addable Ls for prior/marginal probabilities calculation.
        
        Currently rate sampled gamma distribution of given (shape, size).
        """
        
        # check library available:
        if not Ls_library:
            if isinstance(self.Ls_library, dict):
                Ls_library = self.Ls_library
            else:
                raise RuntimeError('Cannot add Lindblad process as process library not specified')
            
        # gather all possible additions, ie. combinations (TLS, L in library but not on TLS)
        # (all treated as equally probable)
        possible_additions = []
        for TLS in model.TLSs:
            addable_Ls = [x for x in Ls_library if x not in TLS.Ls] # addable Ls for this TLS
            possible_additions.extend([(TLS, x) for x in addable_Ls])
        
        # update model if required and additions possible, otherwise leave unchanged:
        if possible_additions:
            # pick one pair of TLS and L operator:
            TLS, operator = possible_additions[np.random.choice(len(possible_additions))]
            
            # sample rate from distribution of type specified here and properties in library
            new_rate = np.random.gamma(*Ls_library[operator])
            
            # update model:
            TLS.Ls[operator] = new_rate
            model.build_operators()
        else:
            pass
        
        # return model and number of possible additions:
        return model, len(possible_additions)
    
    
    
    def add_random_qubit_coupling(self,
                                  model: type(basic_model) | type(learning_model),
                                  qubit_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False
                                  # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                                  ) -> tuple[type(basic_model) | type(learning_model), int]:
        
        """
        Adds random coupling between any qubit and random defect.
        
        Qubit couplings library argument should be dictionary. Keys are:
        tuple of length 2 tuples of labels for operator on one and operator on other subsystem.
        Order shouldn't matter under Hermiticity condition.
        Values are tuples of (shape, scale) of mirrored gamma distribution to sample coupling strength from.
        
        Presently coupling information only stored on one participant TLS to avoid duplication.
        Storage format: 
        {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
        """
        
        # check library available:
        if not qubit_couplings_library:
            if isinstance(self.qubit_couplings_library, dict):
                qubit_couplings_library = self.qubit_couplings_library
            else:
                raise RuntimeError('Cannot add qubit-random defect coupling as library not specified')
        
        # gather all qubit-defect pairs [qubit, defect]:
        pairs = [] 
        for TLS1 in model.TLSs:
            for TLS2 in model.TLSs:
                if TLS1.is_qubit and not TLS2.is_qubit:
                    pairs.append([TLS1, TLS2])
        
        # gather all existing couplings:
        # ie. list of sets each containing {qubit, defect, (op_one, op_other), ...}
        # !!! order in tuples shouldn't matter if the whole thing is hermitian, right?
        existing = []
        for pair in pairs:
            if pair[1] in pair[0].couplings:
                for coupling in pair[0].couplings[pair[1]]:
                    existing.append(set(pair + coupling[1]))
            if pair[0] in pair[1].couplings:
                for coupling in pair[1].couplings[pair[0]]:
                    existing.append(set(pair + coupling[1]))
                    #{partner: [(rate, [(op_self, op_partner), (op_self, op_partner)]]}
                
        # all available couplings from library (even if already present):
        # ie. list of sets each containing {qubit, defect, (op_one, op_other), ..., (strength_shape, strength_scale)}
        available = []
        for pair in pairs:
            for coupling, strength_distribution in qubit_couplings_library.items():
                
                # make into tuple of tuples if passed as ('x','y') or (('x','y')) instead of (('x','y'),)
                if type(coupling) == tuple and list(map(type, coupling)) == [str, str]:
                    coupling = (coupling,)
                available.append(set(list(coupling)+pair+[strength_distribution]))
                
        # complementary list to "available" without strength distribution to enable set comparison with "existing":
        available_comp = []
        for coupling_set in available:
            available_comp.append(
                {x for x in coupling_set if not (type(x) == tuple and set(map(type,x)) <= {int, float})})
                # ie. set of just partners and operator terms
                # sets in same order in list
        
        # gather allowed additions (represented as set) and choose one:
        possible_additions = [x for (x, y) in zip(available, available_comp) if y not in existing]    
        chosen_addition = np.random.choice(possible_additions)
        
        # unpack chosen coupling (set) into TLS identifiers, op label tuples, strength properties tuple:
        new_ops = []
        new_pair = []
        for element in chosen_addition:
            if isinstance(element, tuple):
                if list(map(type, element)) == [str, str]:
                    new_ops.append(element)
                if set(map(type, element)) <= {int, float} and len(element) == 2:
                    new_strength_properties = element
            if isinstance(element, two_level_system.TwoLevelSystem):
                new_pair.append(element) # assumed there will be two matches in set
        
        # format and incorporate new coupling:
        # note: put on 1st TLS in list with 2nd one as partner
        new_strength = np.random.gamma(*new_strength_properties)
        if new_pair[1] not in new_pair[0].couplings:
            new_pair[0].couplings[new_pair[1]] = [] 
        new_pair[0].couplings[new_pair[1]].append((new_strength, new_ops))
        #{partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}        
                
        model.build_operators()
        
        return model, len(possible_additions)
        
        
        
    def add_random_defect2defect_coupling(self,
                                          model: type(basic_model) | type(learning_model),
                                          defect_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False
                                          # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                                          ) -> type(basic_model) | type(learning_model):
        
        """
        Adds random coupling between two random defects.
        
        Currently cannot recognise equivalent couplings if specified using different operators.
        (Possibly to introduce: Could do matrix product comparison,
         would need to run over all couplings instead of succint sets comparison.)
        
        Currently only supports same operator acting on both participants to ensure Hermiticity.
        !!! To do: Extend to multiple operators (eg. sigmap on one sigmam on other plus conjugate).
        Will entail either rewriting how coupling information is stor in model
        so that multiple operators are grouped under single parameter (another layer of list),
        or params handling tweak method to check for related couplings (Hermitian conjugates)
        and only vary their parameter together to preserve Hermiticity.
        
        Presently coupling information only stored on one participant TLS to avoid duplication.
        Storage format: 
        !!! CHANGE {partner : [(strength, op_this, op_partner)]} # partner is object reference and ops label strings
        """
        
        # check library available:
        if not defect_couplings_library:
            if isinstance(self.defect_couplings_library, dict):
                defect_couplings_library = self.defect_couplings_library
            else:
                raise RuntimeError('Cannot add random defects coupling as library not specified')
                
        """     
        1) list of pairs
        2) list of present coupling sets
        3) list of available coupling sets
        4) list of new possibilities
        5) choice one
        6) convert into structured coupling information
        7) update model
        8) return model and count (length of possibilities)
        """
        
        # care below is for qubit-defect
        
            
                
        # all defects:
        defects = [x for x in model.TLSs if not x.is_qubit]
            
        # reapeat up to iterations limit in case randomly chosen coupling already exists
        for i in range(10):
        
            # select first random defect:
            defect1 = np.random.gamma(defects)
            
            # select random partner defect:
            remaining_defects = [x for x in model.TLSs if ((not x.is_qubit) and x != defect1)]
            defect2 = np.random.choice(remaining_defects)
            
            # select random coupling operator and rate bounds from the library:
            operator = np.random.choice([x for x in defect_couplings_library])
            
            # tracker for if any coupling between the two exists
            # note: used to decide whether to append or create new entry in couplings dictionary
            some_coupling = False
            
            # check if this coupling already exists on defect1 - if so try again random choice up to iterations limit:
            if defect2 in [x for x in defect1.couplings]:
                some_coupling = True
                if operator in [y[1] for y in defect1.couplings[defect2]]: 
                # note: checking this against operator on defect but both should be same so far until mixed operators implemented
                    continue
                    
            # likewise check on defect2:
            if defect1 in [x for x in defect2.couplings]:
                
                if operator in [y[1] for y in defect2.couplings[defect1]]:
                # note: checking this against operator on qubit but both should be same so far until mixed operators implemented
                    continue
            
            # ie coupling via this operator between the two not yet existing:
                
            # draw strenght:
            strength = np.random.uniform(*defect_couplings_library[operator])
            
            # initialise list if no coupling between the two exists yet:
            if not some_coupling:
                defect1.couplings[defect2] = []
            
            # add new coupling:
            defect1.couplings[defect2].append((strength, operator, operator))
            model.build_operators()
            
            return model
        
            break # safety break
        
        
        
    def remove_random_L(self,
                        model: type(basic_model.BasicModel) | type(learning_model.LearningModel),
                        update: bool = True
                        ) -> (type(basic_model.BasicModel) | type(learning_model.LearningModel), int):
    
        """
        Can remove random existing single-site Linblad process from random subsystem.
        Modifies argument model and returns: updated model, number of removable Ls.
        Update flag true means removal performed; false avoids changing model,
        to only get count of removable Ls for prior/marginal probabilities calculation.
        """
        
        # gather all possible removals, ie. combinations (TLS, existing L on TLS)
        # (all treated as equally probable)
        possible_removals = []
        for TLS in model.TLSs:
            possible_removals.extend([(TLS, x) for x in TLS.Ls])
        
        # pick one pair of TLS and L operator:
        if possible_removals:
            TLS, operator = possible_removals[np.random.choice(len(possible_removals))]
            
            # update model:
            TLS.Ls.pop(operator)
            model.build_operators()
        else:
            pass
            
        # return model and number of possible removals:
        return model, len(possible_removals)
     
    
        
    def define_Ls_library(self, Ls_library: dict[str, tuple | list]) -> None:
        
        """
        Sets process library post-initialisation.
        """ 
        
        self.Ls_library = Ls_library
        
        
        
    def reset_Ls_library(self) -> None:
        
        """
        Resets Ls library to initial one.   
        """    
    
        self.Ls_library = copy.deepcopy(self.initial_Ls_library)
        
    
    
    def rescale_Ls_library_range(self, factor: int | float) -> None:
    
        """
        Rescales bound distance from middle of range by half of given factor for all process library rates.
        Range hence changes by up to factor.
        Rates assumed positive and capped at zero from left (hence range can change less).
        """
    
        if type(self.Ls_library) == bool and not self.Ls_library:
            raise RuntimeError('Process library not retained by handler so cannot rescale range')
    
        for operator in self.Ls_library:
            # note: bounds are currently tuples and not mutable!
            
            # calculate new bounds:
            old_bounds = self.Ls_library[operator]
            middle = (old_bounds[1] + old_bounds[0])/2
            old_distance = old_bounds[1] - old_bounds[0]
            new_offset = old_distance*factor/2
            new_bounds = (max(0, middle - new_offset), middle + new_offset)
            
            # save in library:
            self.Ls_library[operator] = new_bounds
            
            
            
    def remove_random_qubit_coupling(self, 
                                     model: type(basic_model.BasicModel) | type(learning_model.LearningModel)
                                     ) -> type(basic_model.BasicModel) | type(learning_model.LearningModel):
        """
        Removes random coupling process between qubit and defect.
        """        
            
        # make list of all model's individual couplings provided they are qubit-defect:
        couplings = [] # now tuples of (holding_TLS, partner, index in list)
        for TLS in model.TLSs:
            for partner in TLS.couplings:
                if (TLS.is_qubit and not partner.is_qubit) or (not TLS.is_qubit and partner.is_qubit):
                    for index, coupling in enumerate(TLS.couplings[partner]):
                        couplings.append((TLS, partner, index))
                        
        if (temp := len(couplings) > 0):
            
            # choose coupling to remove if any:
            # note: np.random.choice does not like choosing from list of tuples,
            # ...hence need to choose by list element index 
            selection_index = np.random.choice(temp)
            selection = couplings[selection_index]
            
            # remove coupling:
            selection[0].couplings[selection[1]].pop(selection[2])
            # note: empty list remains if last coupling for given partner
            # hence remove partner from TLS's couplings dictionary:
            if not selection[0].couplings[selection[1]]:
                selection[0].couplings.pop(selection[1])
                
            model.build_operators()  
        
        return model
        
        
            
    def remove_random_defect2defect_coupling(self, 
                                             model: type(basic_model.BasicModel) | type(learning_model.LearningModel)
                                             ) -> type(basic_model.BasicModel) | type(learning_model.LearningModel):
        
        """
        Removes random coupling process between two different defects.    
        """
    
        # make list of all model's individual couplings provided they are defect-defect:
        couplings = [] # now tuples of (holding_TLS, partner, index in list)
        for TLS in model.TLSs:
            for partner in TLS.couplings:
                if (not TLS.is_qubit and not partner.is_qubit):
                    for index, coupling in enumerate(TLS.couplings[partner]):
                        couplings.append((TLS, partner, index))
        
        if (temp := len(couplings) > 0):
            
            # choose coupling to remove if any:
            # note: np.random.choice does not like choosing from list of tuples,
            # ...hence need to choose by list element index 
            selection_index = np.random.choice(temp)
            selection = couplings[selection_index]
            
            # remove coupling:
            selection[0].couplings[selection[1]].pop(selection[2])
            # note: empty list remains if last coupling for given partner
            # hence remove partner from TLS's couplings dictionary:
            if not selection[0].couplings[selection[1]]:
                selection[0].couplings.pop(selection[1])
                
            model.build_operators()  
        
        return model
            
            