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

if typing.TYPE_CHECKING:
    from learning_chain import LearningChain


class ProcessHandler():
    
    """
    Instance provides access to methods adding and removing model processes,
    ie. coupling terms in Hamiltonian or Lindblad operators.
    
    Holds libraries of available processes and corresponding distributions their parameters are sampled from.
    """    
    
    def __init__(self,
                 chain: type(LearningChain) = False,
                 model: type(basic_model) | type(learning_model) = False,
                 Ls_library: dict[str, tuple | list] = False, # {'op label': (shape, scale)}
                 qubit_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False,
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
        
        Currently rate sampled from uniform distribution, bounds given by tuple for each library entry.
        !!! To do: Change this to another distribution with unbounded tails, possibly mirrored gamma.
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
            new_rate = np.random.uniform(*Ls_library[operator])
            
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
                                  ) -> type(basic_model) | type(learning_model):
        
        """
        Adds random coupling between first qubit and random defect.
        
        Currently only supports same operator acting on both participants to ensure Hermiticity.
        
        !!! To do: Extend to multiple operators (eg. sigmap on one sigmam on other plus conjugate).
        Will entail either rewriting how coupling information is stor in model
        so that multiple operators are grouped under single parameter (another layer of list),
        or params handling tweak method to check for related couplings (Hermitian conjugates)
        and only vary their parameter together to preserve Hermiticity.
        
        Presently coupling information only stored on one participant TLS to avoid duplication.
        Storage format: 
        {partner : [(strength, op_this, op_partner)]} # partner is object reference and ops label strings
        """
        
        # check library available:
        if not qubit_couplings_library:
            if isinstance(self.qubit_couplings_library, dict):
                qubit_couplings_library = self.qubit_couplings_library
            else:
                raise RuntimeError('Cannot add qubit-random defect coupling as library not specified')
        
        # first qubit:
        qubit = [x for x in model.TLSs if x.is_qubit][0]
            
        # all defects:
        defects = [x for x in model.TLSs if not x.is_qubit]
        
        
        # reapeat up to iterations limit in case randomly chosen coupling already exists
        for i in range(10):
        
            # select random defect:
            defect = np.random.choice(defects)
            
            # select random coupling operator and rate bounds from the library:
            operator = np.random.choice([x for x in qubit_couplings_library])
            
            # tracker for if any coupling between the two exists
            # note: used to decide whether to append or create new entry in couplings dictionary
            some_coupling = False
            
            # check if this coupling already exists on defect - if so try again random choice up to iterations limit:
            if qubit in [x for x in defect.couplings]:
                
                if operator in [y[1] for y in defect.couplings[qubit]]: 
                # note: checking this against operator on defect but both should be same so far until mixed operators implemented
                    continue
                    
            # likewise check on qubit:
            if defect in [x for x in qubit.couplings]:
                some_coupling = True
                if operator in [y[1] for y in qubit.couplings[defect]]:
                # note: checking this against operator on qubit but both should be same so far until mixed operators implemented
                    continue
            
            
            # ie coupling via this operator between the two not yet existing:
                
            # draw strenght:
            strength = np.random.uniform(*qubit_couplings_library[operator])
            
            # initialise list if no coupling between the two exists yet:
            if not some_coupling:
                qubit.couplings[defect] = []
            
            # add new coupling:
            qubit.couplings[defect].append((strength, operator, operator))
            model.build_operators()
            
            return model
        
            break # safety break
            
        
        
    def add_random_defect2defect_coupling(self,
                                          model: type(basic_model) | type(learning_model),
                                          defect_couplings_library: dict[tuple[tuple[str] | str], tuple[int | float]] = False
                                          # coupling libraries: { ((op_here, op_there), ...) : (shape, scale)}
                                          ) -> type(basic_model) | type(learning_model):
        
        """
        Adds random coupling between two random defects.
        
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
                
        # all defects:
        defects = [x for x in model.TLSs if not x.is_qubit]
            
        # reapeat up to iterations limit in case randomly chosen coupling already exists
        for i in range(10):
        
            # select first random defect:
            defect1 = np.random.choice(defects)
            
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
            
            