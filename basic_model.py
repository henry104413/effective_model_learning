#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


import numpy as np
import scipy as sp
import typing
import qutip

import definitions
import two_level_system

# shorthands:
T = definitions.T
ops = definitions.ops



class BasicModel():
    
    """
    Instance is a model made up of TwoLevelSystem instances (TLSs).
    
    Holds information about constituent TLSs, couplings between them, and Lindblad processes.
    Has methods to generate full system operators and dynamics as per Lindblad Master Equation.
    Also to output model description as a JSON dictionary or string.
    
    Predominantly created by dynamically adding TLSs via add_TLS method.
    Limited random setup also possible.
    
    !!! To add: Initial state specification - now qubits assumed excited and defects ground.
    """
    
    # container for references to all existing Model instances:
    existing_models = []

    
    
    def __init__(self):
        
        # constituent TLSs (instances of TwoLevelSystem):
        self.TLSs = []
        
        # TLSs manual labels - populated if label defined upon TLS creation:
        self.TLS_labels = dict()
        
        # full system Lindblad operators:
        self.Ls = []
        
        # full system Hamiltonian:
        self.H = None
        
        # default full system initial state:
        self.initial_DM = None
        
        # full Liouvillian:
        self.Liouvillian = None
        
        # save instance:
        BasicModel.existing_models.append(self)
        

    
    def add_TLS(self,
                TLS_label: str = '',
                is_qubit: bool = False,
                energy: int|float = None, 
                couplings: dict[two_level_system.TwoLevelSystem, list[tuple[float|int, list[tuple[str, str]]]]] = None,
                Ls: dict[str, int|float] = None,
                initial_state: qutip.Qobj = False
                ) -> two_level_system.TwoLevelSystem:
        
        """
        Initialises new TwoLevelSystem instance with variables set to call arguments.
        Saves reference to this model's container, also returns it,
        additionally saves reference to labels dictionary if label defined.
        """
        
        # process arguments: 
        
        if couplings is None: couplings = {}    
        if Ls is None: Ls = {}
        
        # replace any entries with labels by entries with references):
        # note: TLS objects should only ever have instance references as keys,
        # whereas model instances can handle TLS identifying strings as a user interface
        # note: cannot modify dictionary keys while looping over it with .items()
        for key in list(couplings):
            if type(key) == str:
                couplings[self.TLS_labels[key]] = couplings[key]
                del couplings[key]
                
        # encase in list if any partner's coupling information passed as single tuple not in list      :
        # (ie. input against argument specifications)
        for partner in couplings:
            if type(couplings[partner]) == tuple:
                # assuming otherwise correct input (strength, [(op_here, op_there), ...])
                # alternatively (strength, (op_here, op_there)) would be fixed below
                print('correction...')
                couplings[partner] = [couplings[partner]]    
        
        # encase in list if any coupling's operator-pair passed as single tuple not in list:
        # (ie. input against argument specifications)
        for partner in couplings:
            for i, coupling in enumerate(couplings[partner]):
                if len(coupling) != 2:
                    # note: here coupling assumed a tuple (strength, (op_here, op_there))
                    # false if eg. (strength, op_here, op_there)
                    raise RuntimeError('Incorrect coupling specification:\n'
                                       +'Use {partner: [(rate, [(op_self, op_partner), ...]]}')
                # assuming single operator-pair tuple passed for given strength - encase in list
                if type(coupling[1]) == tuple:
                    couplings[partner][i] = (coupling[0], [coupling[1]])
                    
    
        # make new TLS instance:
        new_TLS = two_level_system.TwoLevelSystem(self, 
                                                  TLS_label, 
                                                  is_qubit, 
                                                  energy, 
                                                  couplings, 
                                                  Ls,
                                                  initial_state)        
        
        # add label to reference dictionary if one is specified
        if TLS_label != '':
            self.TLS_labels[TLS_label] = new_TLS    
        
        # save TLS instance to container:
        self.TLSs.append(new_TLS)
        
        return new_TLS
            
        
      
    def build_operators(self):
        """
        Builds all full system operators - to be called after any changes to model.
        NB: Liouvillian not updated to save costs - call manually if required.
        """
        
        self.build_Ls()
        self.build_H()
        self.build_initial_DM() # note: sometimes not required but included for simplicity as cost fairly low
    
    
    
    def build_Ls(self) -> list[qutip.Qobj]:
        """
        Builds all full system Lindblad operators (dissipators).
        Saves list of all to this model's container, also returns it.
        """
    
        # temporary operators container:
        temp_Ls_container = []
        
        # go through individual TLSs and build all Ls for each:
        for TLS in self.TLSs:
            
            # go through all Ls acting on this TLS and build each:
            for L in TLS.Ls: # note: iterates over just keys
                
                # put operator in correct place and identities in place of other other TLSs:

                # determine first operator in tensor product:    
                # note: required for qutip compatibility -- cannot start with identity/blank object
                if self.TLSs.index(TLS) == 0: # i. e. this is first TLS
                    temp = ops[L]*np.sqrt(TLS.Ls[L])
                else: # else start with identity
                    temp = ops['id2']
            
                # put subsequent identities or this L in correct places: 
                for place in self.TLSs[1:]:
                    if place == TLS:
                        temp = T(temp, ops[L]*np.sqrt(TLS.Ls[L]))
                    else:
                        temp = T(temp, ops['id2'])
                
                # save finished operator:
                # note: now not tracked -- later can be to modify or remove without redoing this
                temp_Ls_container.append(temp)
        
        self.Ls = temp_Ls_container
        
        return temp_Ls_container
        
        
    
    def build_H(self):
        """
        Builds full system Hamiltonian.
        Saves to this model's container, also returns it.
        """

        # set and return None if no TLSs present:
        if not self.TLSs: 
            self.H = None
            return None

        # make energy term for first TLS:
        # note: first term in Hamiltonian required at start as qutip can only add like objects    
        H = ops['sigmaz']*self.TLSs[0].energy
        for i in range(len(self.TLSs[1:])):
            H = T(H, ops['id2'])
            
        # add energy terms for remaining TLSs:
        for TLS in self.TLSs[1:]:
            # build term for this TLS with energy-setting in correct place and identities elsewhere:
            temp = ops['id2']
            for place in self.TLSs[1:]:
                if place == TLS:
                    temp = T(temp, ops['sigmaz']*TLS.energy)
                else:    
                    temp = T(temp, ops['id2'])
            H = H + temp
            
            
        # go over all TLSs and build all its couplings:
        # note: now coupling only recorded within one TLS of each coupled pair
        # assumed structure: {partner: [(rate, [(op_self, op_partner), (op_self, op_partner)]]}
        for TLS in self.TLSs:
            
            # go over all partners to this TLS:
            for partner in TLS.couplings:
                    
                # go over couplings with this parner
                # ie. tuples of (rate, [(op_this, op_partner), (op_this, op_partner), ...]):
                for coupling in TLS.couplings[partner]:
                    
                    # go over all operator combinations sharing same strenght/rate
                    # ie. tuples of (op_on_TLS, op_on_partner):
                    strength, op_pairs = coupling
                    for op_pair in op_pairs:
                        op_on_TLS, op_on_partner = op_pair
                        temp = 1
                        
                        # insert appropriate operator in each place in Kronecker product 
                        for place in self.TLSs:
                            
                            if place == TLS:
                                temp = T(temp, strength*ops[op_on_TLS])
                            elif place == partner:
                                temp = T(temp, strength*ops[op_on_partner])
                            else:
                                temp = T(temp, ops['id2'])
                                
                        H = H + temp
                        
             
        self.H = H
        return H
    
    
    
    def build_initial_DM(self) -> qutip.Qobj:
        """
        Builds default initial state full density matrix.
        Assumes product state with:
        all defects as defined or ground if not,
        all qubits as defined or equal superposition with positive coherence.
        Saves to this model's container, also returns it.
        """
        temp = 1
        for TLS in self.TLSs:
            if TLS.initial_state:
                temp = T(temp, (TLS.initial_state))
                # must be valied initial state
                # don't use unit() of density matrices! it's operator norm, clearly does something else
            else:
                if TLS.is_qubit:
                    temp = T(temp, ops['plus'])
                    # note: default defined here if no other passed
                    # make sure it is physical!!!
                else:
                    temp = T(temp, ops['gnd'])
        self.initial_DM = temp
        return temp
    
    
    
    def build_Liouvillian(self) -> np.ndarray:
        """
        Builds full system Liouvillian as a numpy array.
        Saves to this model's container, also returns it.
        """
        
        # check H available:
        if not self.H:
            raise RuntimeError('Liouvillian cannot be calculated due to missing TLSs or H')
        
        # H and list of Ls as numpy arrays:
        H = np.array(self.H)
        Ls = [np.array(L) for L in self.Ls]
        
        # Hilbert space dimension and corresponding identity as numpy array:
        n = len(H)
        I = np.eye(n)
        
        # Liouvillian:
        temp = -1j*(np.kron(I, H) - np.kron(H.transpose(), I))
        for L in Ls:
            temp += (np.kron(L.conjugate(), L) 
                            - 1/2*(np.kron(I, L.conjugate().transpose()@L) + np.kron(L.transpose()@L, I)))
        
        self.Liouvillian = temp
        return temp
    
        
        
    def calculate_dynamics(self, 
                          evaluation_times: np.ndarray,
                          observable_ops: list[str] = False,
                          # default: qutip -- unless liouvillian==True
                          dynamics_method: str = False,
                          custom_function_on_return: callable = False
                          ) -> list[np.ndarray]:
        
        """
        Calculates dynamics and returns list of arrays of measurements at evaluation_times;
        each array is measurement with respect to corresponding observable argument observable_ops.
        
        If observable not specified, assume population in site basis of first qubit in system.
        If string matching existing operator, take that observable on first qubit found with identities elsewhere.
        
        Custom function acts on output before returning, assumed to take into account output structure.
        
        Note: Apparently evaluation_times needs to be sorted.
        """
    
        # check method argument:
        if dynamics_method == False: dynamics_method = 'qutip'
        if dynamics_method not in ['qutip', 'liouvillian']:
            print('Model dynamics calculation method not recognised, hence using qutip')
            dynamics_method = 'qutip'
        
        # check H and initial state available:
        if self.H == None:
            raise RuntimeError('Hamiltonian not defined -- cannot calculate dynamics')
        if self.initial_DM == None:
            raise RuntimeError('Initial state not defined -- cannot calculate dynamics')
        
        # for observables on single qubit: (either if not specified, a single string, or a list of strings)    
        # make list of total system observable operators:
        # if single TLS operator defined: apply on first qubit in TLSs and identity on others
        
        # not defined hence assume excited population:
        if type(observable_ops) == bool and not observable_ops: observable_ops = 'exc'
        
        # if single operator label (string):
        if isinstance(observable_ops, str): observable_ops = [observable_ops]
        
        # triggered if not valid input (should be list with zero non-string elements):
        if not (isinstance(observable_ops, list) 
                and sum([True for x in observable_ops if type(x) != str]) == 0):
                # !!! note: this can be done more efficienly with a generator
            raise RuntimeError('do not understand oservable operators input for dynamics calculation')
            
        else: 
        # ie. now have list of strings for different observable operators and assuming they match an operator:
        # note: each observable by default applied on first qubit among TLSs 
            observable_ops_full = []
            for op in observable_ops:
                temp = 1
                already_got_one = False
                for TLS in self.TLSs:
                    if TLS.is_qubit and not already_got_one:
                        temp = T(temp, ops[op])
                        already_got_one = True
                    else:
                        temp = T(temp, ops['id2'])
                observable_ops_full.append(temp)
                
        
        # solve ODE using qutip (default option):
        # return list of arrays of observables values across evaluation times, in order of observable operators
        if dynamics_method == 'qutip':
            qutip_dynamics = qutip.mesolve(self.H,
                                     self.initial_DM,
                                     evaluation_times,
                                     c_ops = self.Ls,
                                     e_ops = observable_ops_full,
                                     options = qutip.Options(nsteps = 1e9) # store_states = True # old nsteps = 1e9
                                     # note: creates instance of Options which has as variables all the solver options - can be set in constructor
                                     )
            returnable = qutip_dynamics.expect # alternatively qutip_dynamics.states
            if not custom_function_on_return:
                return returnable 
            else:
                return custom_function_on_return(returnable)
    
        
        # alternative: build Liouvillian and exponentiate - currently deprecated:
        elif dynamics_method == 'liouvillian':
            
            # first of full observables taken - not tested!
            observable_op = np.array(observable_ops_full[0])
                
            
            # vectorised initial DM as numpy array:
            initial_DM_mat = np.array(self.initial_DM)            
            n = len(initial_DM_mat)
            DM_vect = np.reshape(initial_DM_mat, (int(n**2), 1))
            
            LLN = self.build_Liouvillian()
            
            # propagator by time interval:
            # note: assumes evaluation_times evenly spaced!
            dt = evaluation_times[1] - evaluation_times[0]
            # note: diagonalising and exponentiating: cost of diagonalisation comparable to exponentiation
            # also there must be a precision problem -- Liouvillian should be diagonalisable but transforming
            # diagonal matrix of eigenvalues back gives a slightly different matrix and dynamics is divergent
            P = (sp.linalg.expm(dt*LLN))
            self.P = P # save to instance for testing
            
            # propagate and obtain observables at argument times:
            liouvillian_observable = []
            for i in range(len(evaluation_times)):
                if i > 0: DM_vect = P@DM_vect # evolve one time step
                DM_mat = np.reshape(DM_vect, (n, n)) # reshape new DM into matrix
                new_value = np.trace(np.array(observable_op)@DM_mat)
                liouvillian_observable.append(new_value) # extract and save observable
            return liouvillian_observable    
                
        
        
    def disp(self) -> None: 
        """
        Displays full model parameters.
        """
        print(self.model_description_str())
                    
    
    
    def model_description_str(self) -> str:
        "Returns model description as a string (to either print or save to file)."
        outstring = '______Model:______'    
        for TLS in self.TLSs:
            if TLS.is_qubit: outstring += ('\n\n(' + str(self.TLSs.index(TLS)) + ') - Qubit')
            else: outstring += ('\n\n(' + str(self.TLSs.index(TLS)) + ') - Defect')
            outstring += ('\nEnergy: ' + str(TLS.energy))
            outstring += ('\nCouplings:')
            for partner in TLS.couplings: # note: each iterand: partner TLS
                outstring += ('\n   Partner: TLS (' + str(self.TLSs.index(partner)) + ')')
                for coupling in TLS.couplings[partner]: # note: each iterand now (strength, [(op_here, op_there), ...](
                    strength, op_pairs = coupling
                    outstring += ('\n      ' + str(strength) + ':')
                    for op_pair in op_pairs:
                        outstring += ('\n         (' + str(op_pair[0]) + ', ' + str(op_pair[1]) + ')')
            outstring += ('\nLindblad processes:')
            for L in TLS.Ls: # note: each iterand: Lindblad process        
                outstring += ('\n   ' + L + ': ' + str(TLS.Ls[L]))
        outstring += '\n__________________'
        return outstring
    
    
    
    def model_description_dict(self):
        """
        Returns model description as a JSON compatible dictionary.
        """
    
        # make nested dictionary:
        outdict = {}   
        
        # create entry for each constituent TLS:
        for TLS in self.TLSs:
            
            # thsi TLS type and energy:
            subdict = {}
            TLS_id = str(self.TLSs.index(TLS))
            if TLS.is_qubit: subdict['type'] = 'Qubit'
            else: subdict['type'] = 'Defect'
            subdict['energy'] = TLS.energy
        
            # this TLS couplings information:
            subsubdict = {}
            for partner in TLS.couplings: # note: each iterand: partner TLS
                subsubdict[str(self.TLSs.index(partner))] = (TLS.couplings[partner])
                # for coupling in TLS.couplings[partner]: # note: each iterand: tuple of individual coupling term details 
                #     outstring += ('\n      (' + str(coupling[0]) + ', ' + (coupling[1]) + ', ' + (coupling[2]) + ')')
            subdict['couplings:'] = subsubdict

            # this TLS Lindblads information:
            subsubdict = {}
            for L in TLS.Ls: # note: each iterand: Lindblad process        
                subsubdict[L] = TLS.Ls[L]
            subdict['Ls:'] = subsubdict
            
            outdict[TLS_id] = subdict
                    
        return outdict
    
    
    
    def random_setup(self,
                     qubits_number: int = 1, # now assumed one
                     defects_number_range: tuple[int] = (2,2), # range given by two-element tuple
                     qubits_energies: list[int, float] = [5], # list of floats the length of qubits_number - assumed known
                     defects_energies_range: tuple[int, float]  = (3, 7), # tuple of two floats for range
                     allowed_couplings: dict = {
                         # identifier: now should be qubit due to restrictions
                         'qubit': [(1, (0.1, 1) , [('sigmax', 'sigmax')])] # partner: [(probability, (rate min, rate max), [type])]
                         #...list for different couplings to this partner
                         },
                     allowed_Ls: dict[str, tuple[int|float, tuple[int|float]]] = {
                         'sigmaz': (1, (0.005, 0.1)) # type: (probability, rate range)
                         }
                     ) -> None:
    
        """
        Randomly populates model.
        Warning: Calling this replaces any previous configuration.
        Now assuming 1 qubit and couplings only between it and single defects.
        
        Method not fully maintained.
        """
        
        # remove old constituents: (operators get replaced by #calling build_operators() later)
        self.TLSs = []
        self.TLS_labels = {}
                   
        # returns dictionary of Ls generated according to input to configure single TLS:                                      
        def get_Ls(): 
            Ls = {}
            for L, val in allowed_Ls.items(): 
            # note: val[0] is probability, val[1] is range given by tuple - unpackl using *val[1]
                # decide if included based on probability:
                if not np.random.uniform() < val[0]: # ie. allowed_Ls[key] NOT included
                    continue
                else: # ie. included
                    Ls[L] = np.random.uniform(*val[1]) 
            return Ls
        
        # returns dictionary of couplings generated according to input to configure single TLS:                                      
        def get_couplings():
            couplings = {}
            for partner, val in allowed_couplings.items():
            # for this partner now go over all possible couplings:
                complete = []
                for coupling in val:
                    # decide if included based on probability:        
                    if not np.random.uniform() < coupling[0]: # ie. NOT included 
                        continue
                    else: # ie. included
                        complete.append((np.random.uniform(*coupling[1]), coupling[2]))
                couplings[partner] = complete
            return couplings
            
        # make one qubit: (assumed to only be one)        
        self.add_TLS(TLS_label = 'qubit',
                     is_qubit = True,
                     energy = qubits_energies[0],
                     couplings = {},
                     Ls = get_Ls()
                     )
        
        # make defects:
        defects_number = np.random.randint(defects_number_range[0], defects_number_range[1]+1)
        for i in range(defects_number):
            self.add_TLS(
                          is_qubit = False,
                          energy = np.random.uniform(*defects_energies_range),
                          couplings = get_couplings(),
                          Ls = get_Ls()
                          )
        # build operators: 
        self.build_operators()
       
        
       
    def vectorise_under_library(self, hyperparameters:dict, model:typing.Self = False
                                ) -> (list[float], list[str], list[str]):
        """
        Returns tuple of lists: values, shorthand labels, latex labels, 
        with entry for each possible parameter given library of available processes.
        (Values is fixed-length vectorised representation of model,
         with labels indicating what process each element belongs to.)
        
        Argument hyperparameters should be dictionary containing following entries (string keys):
            qubit2defect_couplings_library
            defect2defect_couplings_library
            qubit_Ls_library
            defect_Ls_library
        
        Couplings libraries' keys should be tuples of tuples of operator strings
        (multiple tuples of operators admissible under one coupling),
        Ls libraries' keys should be single strings.
            
        Argument model is any instance of this class, or if not passed, this (calling) instance is taken. 
        
        For each process from library, value is set to corresponding parameter if process found in model,
        or else to zero if absent.
        
        Order of parameters:
            energy splittings (Hamiltonian sigma z term) in order of defects
            qubit-defect (aka s2v) couplings in order of pairs and then operators
            defect-defect couplings (aka v2v) in order of pairs and then operators
            qubit (aka syst) Ls in order of qubits and then operators
            defect (aka virt) Ls in order of defects and then operators
        """
        
        if not model: model = self
        
        def make_op_label(op: str, tex: bool = False)->str:
            """
            For select operator name strings returns a shorthand or latex symbol if tex passed as true.
            """
            match op:
                case 'sigmax':
                    if tex: return r'$\sigma_x$'
                    else: return 'sx'
                case 'sigmay':
                    if tex: return r'$\sigma_y$'
                    else: return 'sy'
                case 'sigmaz':
                    if tex: return r'$\sigma_z$'
                    else: return 'sz'
                case 'sigmap':
                    if tex: return r'$\sigma_+$'
                    else: return 'sp'
                case 'sigmam':
                    if tex: return r'$\sigma_-$'
                    else: return 'sm'


        # gather all qubits, all defects, qubit-defect pairs, and defect-defect pairs,
        # as well as corresponding labels using the system (S) and virtual systems (V) nomenclature

        qubits = [x for x in model.TLSs if x.is_qubit]
        qubits_labels = ['S' + str(i+1) for i in range(len(qubits))] 

        defects = [x for x in model.TLSs if not x.is_qubit]
        defects_labels = ['V' + str(i+1) for i in range(len(defects))]
          
        q2d_pairs = [(q, d) for q in qubits for d in defects]
        q2d_pairs_labels = [qubits_labels[qubits.index(q)] + ',' + defects_labels[defects.index(d)]
                            for q, d in q2d_pairs] 

        d2d_pairs = []
        for d1 in defects:
            for d2 in defects:
                if d1 != d2 and (d2, d1) not in d2d_pairs:
                    d2d_pairs.append((d1, d2))
        d2d_pairs_labels = [defects_labels[defects.index(d1)] + ',' + defects_labels[defects.index(d2)]
                            for d1, d2 in d2d_pairs] 


        # initialise lists for labels, labels with latex symbols, and parameter values
        labels = []
        labels_latex = []
        values = []
            
        
        # populate all:
        
        # defect splittings:
        for defect, label in zip(defects, defects_labels):
            labels.append(label + '-E-')
            labels_latex.append(label + r'$\ E$')
            values.append(defect.energy)
            
        # q2d in order of pairs and then library operators:
        # note: coupling information stored on one of the pair hence check both! 
        for pair, pair_label in zip(q2d_pairs, q2d_pairs_labels):
            for coupling in list(hyperparameters['qubit2defect_couplings_library'].keys()):
                # go over all types of coupling in library, check if present on either of this pair with other as partner:
                # note: learning methods should allow it to only exist on one of pair at a time,
                # otherwise it would be duplicate... 
                # but for completenes and assuming linearity can just add together all rates potential instances of that coupling
                
                ops_label = ''
                ops_label_latex = ''
                for i, op_pair in enumerate(coupling): # coupling is TUPLE of tuples!!
                    if i == 0: ops_label_latex += ' '
                    if i>0: ops_label_latex += '+'
                    ops_label += '-' + make_op_label(op_pair[0]) + ',' + make_op_label(op_pair[1])
                    ops_label_latex += make_op_label(op_pair[0], tex=True) + r'$\otimes$' + make_op_label(op_pair[1], tex=True)
                
                # labels for this process between these pairs
                labels.append(pair_label + ops_label)
                labels_latex.append(pair_label + ops_label_latex)
                
                # note: couplings stored as:
                # TLS.couplings = {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
                
                # check if exists, then populate rate, otherwise leave as zero:
                value = float(0)
                for TLS1, TLS2 in [pair, tuple(reversed(pair))]:
                    # note: takes TLS1 as either of the pair with TLS2 being the other to avoid repetition
                    if TLS2 in TLS1.couplings:
                        for existing_coupling in TLS1.couplings[TLS2]:
                            # go over all existing ones
                            # if matching one from library ie. "coupling", add its rate to value
                            # existing_coupling is tuple (rate, [(op_self, op_partner), (op_self, op_partner), ...])
                            # coupling is tuple of tuples for operators (from library)
                            if set(existing_coupling[1]) == set(coupling): # ie. type is present
                                value += existing_coupling[0]
                values.append(value)
                
        # d2d in order of pairs and then library operators:
        # note: coupling information stored on one of the pair hence check both! 
        for pair, pair_label in zip(d2d_pairs, d2d_pairs_labels):
            for coupling in list(hyperparameters['defect2defect_couplings_library'].keys()):
                # go over all types of coupling in library, check if present on either of this pair with other as partner:
                # note: learning methods should allow it to only exist on one of pair at a time,
                # otherwise it would be duplicate... 
                # but for completenes and assuming linearity can just add together all rates potential instances of that coupling
                
                ops_label = ''
                ops_label_latex = ''
                for i, op_pair in enumerate(coupling): # coupling is TUPLE of tuples!!
                    if i == 0: ops_label_latex += ' '
                    if i>0: ops_label_latex += '+'
                    ops_label += '-' + make_op_label(op_pair[0]) + ',' + make_op_label(op_pair[1])
                    ops_label_latex += make_op_label(op_pair[0], tex=True) + r'$\otimes$' + make_op_label(op_pair[1], tex=True)
                
                # labels for this process between these pairs
                labels.append(pair_label + ops_label)
                labels_latex.append(pair_label + ops_label_latex)
                
                # note: couplings stored as:
                # TLS.couplings = {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
                
                # check if exists, then populate rate, otherwise leave as zero:
                value = float(0)
                for TLS1, TLS2 in [pair, tuple(reversed(pair))]:
                    # note: takes TLS1 as either of the pair with TLS2 being the other to avoid repetition
                    if TLS2 in TLS1.couplings:
                        for existing_coupling in TLS1.couplings[TLS2]:
                            # go over all existing ones
                            # if matching one from library ie. "coupling", add its rate to value
                            # existing_coupling is tuple (rate, [(op_self, op_partner), (op_self, op_partner), ...])
                            # coupling is tuple of tuples for operators (from library)
                            if set(existing_coupling[1]) == set(coupling): # ie. type is present
                                value += existing_coupling[0]
                values.append(value)
         
        # qubit Ls in order of qubits and then ops from library:
        # note: only assuming one value of each type present
        for qubit, qubit_label in zip(qubits, qubits_labels):
            for L in list(hyperparameters['qubit_Ls_library'].keys()):
                labels.append(qubit_label + '-' + make_op_label(L) + '-')
                labels_latex.append(qubit_label + r'$\ $' + make_op_label(L, tex=True))
                if L in qubit.Ls:
                    values.append(qubit.Ls[L])
                else:
                    values.append(float(0))
                
        # defect Ls in order of defects and then ops from library:
        # note: only assuming one value of each type present
        for defect, defect_label in zip(defects, defects_labels):
            for L in list(hyperparameters['defect_Ls_library'].keys()):
                labels.append(defect_label + '-' + make_op_label(L) + '-')
                labels_latex.append(defect_label + r'$\ $' + make_op_label(L, tex=True))
                if L in defect.Ls:
                    values.append(defect.Ls[L])
                else:
                    values.append(float(0))
                    
        
        return values, labels, labels_latex            
            
         
        
        
        
    
            