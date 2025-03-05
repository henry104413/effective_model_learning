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


# common types:
TYPE_COUPLING_LIBRARY = dict[tuple[tuple[str] | str], tuple[int | float]]
TYPE_LS_LIBRARY = dict[str, tuple[int | float]]
TYPE_MODEL = type(basic_model.BasicModel) | type(learning_model.LearningModel)

class ProcessHandler:
    
    """
    Instance provides access to methods adding and removing model processes,
    ie. coupling terms in Hamiltonian or Lindblad operators.
    
    Holds libraries of available processes and corresponding distributions their parameters are sampled from.
    """    
    
    def __init__(self,
                 chain: type(LearningChain) = False,
                 model: TYPE_MODEL = False,
                 Ls_library: TYPE_LS_LIBRARY = False, # {'op label': (shape, scale)}
                 qubit2defect_couplings_library: TYPE_COUPLING_LIBRARY = False,
                 defect2defect_couplings_library: TYPE_COUPLING_LIBRARY = False
                 # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                 ):

        self.Ls_library = Ls_library
        self.initial_Ls_library = copy.deepcopy(Ls_library)
        self.qubit2defect_couplings_library = qubit2defect_couplings_library
        self.defect2defect_couplings_library = defect2defect_couplings_library
        self.model = model # only for future methods tied to specific model
        self.chain = chain # only for future methods tied to chain (eg using its cost function)

        

    def add_random_L(self,
                     model: TYPE_MODEL,
                     Ls_library: TYPE_LS_LIBRARY = False, # {'op label': (shape, scale)}
                     update: bool = True
                     ) -> tuple[TYPE_MODEL, int]:
        
        """
        Can add random new single-site Linblad process from process library to random subsystem.
        Modifies argument model, also returns it as (updated model, # possible additions).
                                                     
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
        if possible_additions and update:
            # pick one pair of TLS and L operator:
            TLS, operator = possible_additions[np.random.choice(len(possible_additions))]
            
            # sample rate from distribution of type specified here and properties in library
            new_rate = np.random.gamma(*Ls_library[operator])
            
            # update model:
            TLS.Ls[operator] = new_rate
            model.build_operators()
        
        # return model and number of possible additions:
        return model, len(possible_additions)
    
    
    
    def add_random_qubit2defect_coupling(self,
                                  model: TYPE_MODEL,
                                  qubit2defect_couplings_library: TYPE_COUPLING_LIBRARY = False,
                                  # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                                  update: bool = True
                                  ) -> tuple[TYPE_MODEL, int]:
        
        """
        Adds random coupling between random qubit and random defect.
        Modifies argument model, also returns it as (updated model, # possible additions).
        Update flag true means addition performed; false avoids changing model.                                             
                                                     
        Qubit couplings library argument should be dictionary. Keys are:
        tuple of length 2 tuples of labels for operator on one and operator on other subsystem.
        Order shouldn't matter under Hermiticity condition.
        Values are tuples of (shape, scale) of mirrored gamma distribution to sample coupling strength from.
        
        Presently coupling information only stored on one participant TLS to avoid duplication.
        Storage format: 
        {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
                    
        Currently cannot recognise equivalent couplings if specified using different operators.
        (Possibly to introduce: Could do matrix product comparison,
         would need to run over all couplings instead of succint sets comparison.)
        """
        
        # check library available:
        if not qubit2defect_couplings_library:
            if isinstance(self.qubit2defect_couplings_library, dict):
                qubit2defect_couplings_library = self.qubit2defect_couplings_library
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
            for coupling, strength_distribution in qubit2defect_couplings_library.items():
                
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
        
        # gather allowed additions (represented as set):
        possible_additions = [x for (x, y) in zip(available, available_comp) if y not in existing]    
        
        # if new coupling available and update flag on:
        if possible_additions and update:
            
            # choose and unpack into TLS identifiers, op label tuples, strength properties tuple:
            chosen_addition = np.random.choice(possible_additions)
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
            
            # format and incorporate into model:
            # note: put on 1st TLS in list with 2nd one as partner
            new_strength = np.random.gamma(*new_strength_properties)
            if new_pair[1] not in new_pair[0].couplings:
                new_pair[0].couplings[new_pair[1]] = [] 
            new_pair[0].couplings[new_pair[1]].append((new_strength, new_ops))
            #{partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}        
            model.build_operators()
            
        return model, len(possible_additions)
        
        
        
    def add_random_defect2defect_coupling(self,
                                          model: TYPE_MODEL,
                                          defect2defect_couplings_library: TYPE_COUPLING_LIBRARY = False,
                                          # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                                          update: bool = True
                                          ) -> tuple[TYPE_MODEL, int]:
        
        """
        Adds random coupling between two random defects.
        Modifies argument model, also returns it as (updated model, # possible additions).
        Update flag true means addition performed; false avoids changing model.
                                                     
        Couplings library argument should be dictionary. Keys are:
        tuple of length 2 tuples of labels for operator on one and operator on other subsystem.
        Order shouldn't matter under Hermiticity condition.
        Values are tuples of (shape, scale) of mirrored gamma distribution to sample coupling strength from.
        
        Presently coupling information only stored on one participant TLS to avoid duplication.
        Storage format: 
        {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
                    
        Currently cannot recognise equivalent couplings if specified using different operators.
        (Possibly to introduce: Could do matrix product comparison,
         would need to run over all couplings instead of succint sets comparison.)
        """
        
        # check library available:
        if not defect2defect_couplings_library:
            if isinstance(self.defect2defect_couplings_library, dict):
                defect2defect_couplings_library = self.defect2defect_couplings_library
            else:
                raise RuntimeError('Cannot add random defects coupling as library not specified')
        
        # gather all pairs [defect1, defect2] without double counting:
        pairs = [] 
        for TLS1 in model.TLSs:
            for TLS2 in model.TLSs:
                if (not TLS1.is_qubit and not TLS2.is_qubit 
                    and ([TLS2, TLS1] not in pairs) and not (TLS1 == TLS2)):
                    pairs.append([TLS1, TLS2])
        
        # gather all existing couplings:
        # ie. list of sets each containing {defect1, defect2, (op_one, op_other), ...}
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
        # ie. list of sets each containing {defect1, defect2, (op_one, op_other), ..., (strength_shape, strength_scale)}
        available = []
        for pair in pairs:
            for coupling, strength_distribution in defect2defect_couplings_library.items():
                
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
        
        # if new coupling available and update flag on:
        if possible_additions and update:
            
            # choose and unpack into TLS identifiers, op label tuples, strength properties tuple:
            chosen_addition = np.random.choice(possible_additions)
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
            
            # format and incorporate into model:
            # note: put on 1st TLS in list with 2nd one as partner
            new_strength = np.random.gamma(*new_strength_properties)
            if new_pair[1] not in new_pair[0].couplings:
                new_pair[0].couplings[new_pair[1]] = [] 
            new_pair[0].couplings[new_pair[1]].append((new_strength, new_ops))
            #{partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}        
                    
            model.build_operators()
            
        return model, len(possible_additions)
        
        
        
    def remove_random_L(self,
                        model: TYPE_MODEL,
                        update: bool = True
                        ) -> tuple[TYPE_MODEL, int]:
    
        """
        Can remove random existing single-site Linblad process from random subsystem.
        Modifies argument model, also returns it as (updated model, # possible removals).
        Update flag true means removal performed; false avoids changing model,
        to only get count of removable Ls for prior/marginal probabilities calculation.
        """
        
        # gather all possible removals, ie. combinations (TLS, existing L on TLS)
        # (all treated as equally probable)
        possible_removals = []
        for TLS in model.TLSs:
            possible_removals.extend([(TLS, x) for x in TLS.Ls])
        
        # pick one pair of TLS and L operator if removals possible and update flag on:
        if possible_removals and update:
            TLS, operator = possible_removals[np.random.choice(len(possible_removals))]
            
            # update model:
            TLS.Ls.pop(operator)
            model.build_operators()
        else:
            pass
            
        # return model and number of possible removals:
        return model, len(possible_removals)
     
    
        
    def define_Ls_library(self, Ls_library: TYPE_LS_LIBRARY) -> None:
        
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
            
            
            
    def remove_random_qubit2defect_coupling(self, 
                                     model: TYPE_MODEL,
                                     update: bool = True
                                     ) -> TYPE_MODEL:
        """
        Removes random coupling process between qubit and defect.
        Modifies argument model, also returns it as (updated model, # possible removals).
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
                                             model: TYPE_MODEL,
                                             update: bool = True
                                             ) -> TYPE_MODEL:
        
        """
        Removes random coupling process between two different defects.    
        Modifies argument model, also returns it as (updated model, # possible removals).
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
            
            