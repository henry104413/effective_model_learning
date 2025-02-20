#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from basic_model import BasicModel
    from learning_model import LearningModel
# note: terrible business... cyclical imports when type checking...
# fix: import only when typing.TYPE_CHECKING AND a) replace dependent types by strings ("forward reference")
# ... OR b) at start of file: from __future__ import annotations (to not have to wrap types into strings for runtime unimported stuff)
# also within TYPE_CHEKCING maybe don't do the full import but only from ... import ... - should be cheaper

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
    
    def __init__(self, model: type(BasicModel) | type(LearningModel),
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
                 
                 
                 