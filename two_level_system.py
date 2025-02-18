#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import basic_model
import learning_model
import typing


class TwoLevelSystem():
    
    """
    Instance is a single two-level system, always belonging to some model. 
    
    Initialiser arguments: parent model, optional identifier label,
    qubit flag (system of interest vs defect), energy (splitting),
    coherent couplings dictionary (form explained below), 
    Linbdlad operators dictionary 
    
    Coherent couplings specification:
    
    old: {partner: [(rate, op on self, op on partner)]}
    new: {partner: [(rate, [(op_self, op_partner), (op_self, op_partner)]]}
    where the two tuples of ops should be Hermitian-conjugate
    
    NB:
    Currently coupling information only stored on one partner.
    Ensures no duplicate entries in Hamiltonians with current model methods implementation.         
      
    """
    
    def __init__(self, model: type(basic_model.BasicModel) | type(learning_model.LearningModel),
                 TLS_id: str = "",
                 is_qubit: bool = False,
                 energy: int|float = None,
                 couplings: dict[typing.Self, list[tuple[float|int, list[tuple[str, str]]]]] = {},
                 # {partner: [(rate, [(op_on_self, op_on_partner)])]}
                 # inner list: all two-site operators in it share the same rate
                 Ls: dict[str, int|float] = {}
                 ):
        
        
        self.model = model
        self.TLS_id = TLS_id
        self.is_qubit = is_qubit
        self.energy = energy
        self.couplings = couplings
        self.Ls = Ls
                 
                 
                 