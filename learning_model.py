#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


from __future__ import annotations
import basic_model
import numpy as np
import configs
import qutip


# enhanced model with ability to modify itself
# methods to change parameters, methods to add/remove processes, and methods to add/remove subsystems

class LearningModel(basic_model.BasicModel):
    
    """
    Instances based on BasicModel (hold information about systems, couplings, processes;
    able to generate full operators and dynamics accodring to Lindblad Master Equation).
    Additionally implements method to shift all parameters by randomly sampled amount,
    with width for each type of parameter set for instance as jump lengths hyperparameter.
    """    
    
    def __init__(self, *args, 
                 # initial_guess = False, 
                 jump_lengths: dict[str, float|int] = False,
                 **kwargs):
        
        # parent initialiser sets up model parameters:
        # note: often initialised empty and components added dynamically 
        super().__init__(*args, **kwargs)
        
        # dictionary of standard jump lengths for each type of operator
        # note: modified with direct method or with argument to parameter changing method
        self.jump_lengths = jump_lengths
        
        # container for final/best loss function:
        # note: only makes sense with respect to some target data,
        # populated by learning chain run only,
        # used in clustering when target known, otherwise can be disregarded
        self.final_loss = False
        
        
    
    def set_jump_lengths(self, passed_jump_lengths: dict[str, float|int] = False) -> None:
        
        """
        Sets jump lengths according to argument dictionary.
        """
        
        self.jump_lengths = passed_jump_lengths
        
    
    
    def old_change_params(self, passed_jump_lengths: dict[str, float|int] = False,
                      bounds: dict[str, tuple[float|int]] = False
                      ) -> None:
        
        """
        Legacy version before implementing rejection sampling with bounds.
        
        Changes all existing parameters of this model except qubit energies,
        each by amount from normal distribution around zero,
        with variance given for each class of parameters (current split: energy, couplings, Ls)
        by jump lengths dictionary if passed or instance variable if not.
        
        Rejection sampling performed if outside of bounds specified as tuples for each parameter class,
        if False then only performed to avoid negative Lindblad rates.
        """
        
        # check jump lengths specified:
        if (type(passed_jump_lengths) == bool and not passed_jump_lengths):
            if (type(self.jump_lengths) == bool and not self.jump_lengths):
                raise RuntimeError('need to specify jump lengths for the operators present')
        else:
            self.jump_lengths = passed_jump_lengths
               
        # take each TLS:    
        for TLS in self.TLSs:
                    
            # modify its energy if not qubit:
            if not TLS.is_qubit:
                TLS.energy += np.random.normal(0, self.jump_lengths['energies'])
            
            # modify all its couplings to each partner:
            # {partner: [(rate, [(op_on_self, op_on_partner)])]}
            for partner in TLS.couplings: # partner is key and value is list of tuples
                this_partner_couplings = TLS.couplings[partner] # list of couplings to current partner
                for i, coupling in enumerate(TLS.couplings[partner]): # coupling now (rate, [(op_on_self, op_on_partner), ...])
                    strength, op_pairs = coupling
                    TLS.couplings[partner][i] = (strength + np.random.normal(0, self.jump_lengths['couplings']), op_pairs)
            
            # modify all its Lindblad ops:
            for L in TLS.Ls:
                # make up to specified number of proposals ensuring result positive
                max_attempts = 10
                for _ in range(max_attempts):    
                    proposal = TLS.Ls[L] + np.random.normal(0, self.jump_lengths['Ls'])
                    if proposal >= 0:
                        TLS.Ls[L] = proposal
                        break
                 
        # remake operators (to update with new parameters):
        self.build_operators()
        
    
        
    def change_params(self, passed_jump_lengths: dict[str, float|int] = False,
                      bounds: dict[str, tuple[float|int]] = False
                      ) -> None:
        
        """
        Changes all existing parameters of this model except qubit energies,
        each by amount from normal distribution around zero,
        with variance given for each class of parameters (current split: energy, couplings, Ls)
        by jump lengths dictionary if passed or instance variable if not.
        
        Rejection sampling performed if outside of bounds specified as tuples for each parameter class,
        if False then only performed to avoid negative Lindblad rates.
        """
        
        # check jump lengths specified:
        if (type(passed_jump_lengths) == bool and not passed_jump_lengths):
            if (type(self.jump_lengths) == bool and not self.jump_lengths):
                raise RuntimeError('need to specify jump lengths for the operators present')
        else:
            self.jump_lengths = passed_jump_lengths
        
        # set max attempts for rejection sampling and infinite bounds if none set to enable comparison:
        # (except  Ls >= 0)
        # (alternatively TO DO: can implement conditional for bounds checking rather than this shortcut)
        max_attempts = 10
        if not bounds: # type validity otherwise not checked
            #print('Bounds not passed to learning model')        
            bounds = {'energies': (-np.inf, np.inf),
                      'couplings': (-np.inf, np.inf),
                      'Ls': (0, np.inf)}
            
        # take each TLS:    
        for TLS in self.TLSs:
                    
            # modify its energy if not qubit:
            if not TLS.is_qubit:
                for _ in range(max_attempts):
                    candidate = TLS.energy + np.random.normal(0, self.jump_lengths['energies'])
                    if candidate >= bounds['energies'][0] and candidate <= bounds['energies'][1]:
                        # within bounds hence update
                        TLS.energy = candidate
                        #print('Energy: accepted ' + str(candidate))
                        break
                    else:
                        #print('Energy: rejected ' + str(candidate))
                        continue
                
            # modify all its couplings to each partner:
            # {partner: [(rate, [(op_on_self, op_on_partner)])]}
            for partner in TLS.couplings: # partner is key and value is list of tuples
                this_partner_couplings = TLS.couplings[partner] # list of couplings to current partner
                for i, coupling in enumerate(TLS.couplings[partner]): # coupling now (rate, [(op_on_self, op_on_partner), ...])
                    strength, op_pairs = coupling
                    for _ in range(max_attempts):
                        candidate = strength + np.random.normal(0, self.jump_lengths['couplings'])
                        if candidate >= bounds['couplings'][0] and candidate <= bounds['couplings'][1]:
                            # within bounds hence update
                            # !!! note: currently disallows negative coupling
                            # ...allowing jumps into interval negative mirror image causes breakdown of 
                            # truncated normal distribution formulas
                            # (but could potentially introduce dedicated negative only couplings etc.) 
                            TLS.couplings[partner][i] = (candidate, op_pairs)
                            #print('Coupling: accepted ' + str(candidate))
                            break
                        else:
                            #print('Coupling: rejected ' + str(candidate))
                            continue
            
            # modify all its Lindblad ops:
            for L in TLS.Ls:
                # make up to specified number of proposals ensuring result positive
                for _ in range(max_attempts):    
                    candidate = TLS.Ls[L] + np.random.normal(0, self.jump_lengths['Ls'])
                    if candidate >= bounds['Ls'][0] and candidate <= bounds['Ls'][1]:
                        TLS.Ls[L] = candidate
                        #print('L: accepted ' + str(candidate))
                        break
                    else:
                        #print('L: rejected ' + str(candidate))
                        continue
                    
        # remake operators (to update with new parameters):
        self.build_operators()
        
        
        
    def configure_to_params_vector(self,
                                   vectorised_model: tuple[list[float|int], list[str], list[str]],
                                   D: int = 2,
                                   default_qubit_energy: float|int = 1,
                                   qubit_initial_state: qutip.Qobj = False,
                                   defect_initial_state: qutip.Qobj = False
                                   # note: initial states are separable DMs for each constituent TLSs
                                   #,config_name: str = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
                                   ) -> LearningModel:
        
        """
        Returns learning model instance populated according to argument vectorised model.
        
        Vectorised model is a tuple of corresponding lists of parameter values, parameter labels, 
        and latex-formatted paremeter labels.
        
        D is defects number, default_qubit_energy is starting qubit energy, assuming one qubit only.
        
        Currently assuming simple model structure with couplings only as single pairs of single operators,
        and assuming all present operators are included in ad hoc operator label conversion dictionary.
        
        TO DO: to be expanded to allow un-vectorising according to hyperparameters specified by configs file
        under currently disabled config_name, which will specify the library.
        The couplings builder should then break the operator pairs string 
        into an arbitrary number of operators present.
        """

        #hyperparams = configs.get_hyperparams(config_name)
        op_short2long = {'sx': 'sigmax', 'sy': 'sigmay', 'sz':'sigmaz', 'sp':'sigmap', 'sm':'sigmam'}
        vals, labels, latex_labels = vectorised_model
        
        # treat this model instance as new model to be emptied and freshly populated:
        # !!! warning: erases old state
        new_model = self
        new_model.TLSs = []
        new_model.add_TLS(is_qubit = True, energy = default_qubit_energy, initial_state = qubit_initial_state)
        for i in range(D):
            new_model.add_TLS(is_qubit = False, initial_state = defect_initial_state) 
        qubits = [x for x in new_model.TLSs if x.is_qubit]
        defects = [x for x in new_model.TLSs if not x.is_qubit]

        # populate model using vectorised parameters and labels:
        # !!! note: currently assumes simple models with a single operator pair for couplings - can be extended
        for label, val in zip(labels, vals):
            if val == 0: continue # skip zero-value parameters
            if (temp := len([True for x in label if x in ['S','V']])) == 2: # ie. coupling
                systems_labels, op1, op2 = label.split('-',3)
                holder_label, partner_label = systems_labels.split(',')
                if holder_label[0] == 'V':
                    holder = defects[int(holder_label[1]) - 1]
                elif holder_label[0] == 'S':
                    holder = qubits[int(holder_label[1]) - 1]
                if partner_label[0] == 'V':
                    partner = defects[int(partner_label[1]) - 1]
                elif partner_label[0] == 'S':
                    partner = qubits[int(partner_label[1]) - 1]
                if partner not in holder.couplings:
                    holder.couplings[partner] = []
                holder.couplings[partner].append((val,[(op_short2long[op1],op_short2long[op2])]))
                # add to dictionary entry DO NOT REPLACE!!
            elif temp == 1 and 'E' in label: # ie. defect energy term 
            # note: qubits were initialised to default but can be changed
                system_label, _ = label.split('-',1)
                if system_label[0] == 'V':
                    system = defects[int(system_label[1]) - 1]
                elif system_label[0] == 'S':
                    system = qubits[int(system_label[1]) - 1]
                system.energy = val
            else: # ie. lindblad
                system_label, op = label.split('-',2)[0:2]
                if system_label[0] == 'V':
                    system = defects[int(system_label[1]) - 1]
                elif system_label[0] == 'S':
                    system = qubits[int(system_label[1]) - 1]
                system.Ls[op_short2long[op]] = val
        new_model.build_operators()

        return new_model             

    
    
    
    

        

            
        
        
        