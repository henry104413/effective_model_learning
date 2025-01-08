#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""



from definitions import T, ops

from two_level_system import TwoLevelSystem

import qutip

import numpy as np

import scipy as sp

import time



class Model():

    

    
    # container for references to all existing Model instances:
    
    existing_models = []
    
    
    
    
    # constructs a new Model instance:
    
    def __init__(self):
        
        
        
        # add: input check for qubit and defects dictionaries
        
        # add: initial state specification - so far assume qubit excited, all else ground
        
        
        
        # containers for reference to this model's qubit TLS instance and list off all Defect instances:
        
        self.TLSs = []
        
        
        # container for manual labels for TLS -- entry added if label defined upon TLS creation
        
        self.TLS_labels = {}
        
        
        # container for this model's Lindblad operators:
        
        self.Ls = []
        
        
        # full model Hamiltonian:
        
        self.H = None
        
        
        # default system initial state:
            
        self.initial_DM = None
        
        
        # save this instance to existing models:
        
        Model.existing_models.append(self)
        
    
    
    
    # adds a TLS to this model with given specifications
    
    def add_TLS(self, TLS_label = '', is_qubit = False, energy = None, couplings = {}, Ls = {}):
        
        
    
        # HERE:
        # process input dictionary, replacing any entries with LABELS with entries with references
        # note: TLS objects should only ever have instance references as keys,
        # ...whereas model instances can handle TLS identifying strings as a user interface
        # note: cannot modify dictionary keys while looping over it with .items()
                
        for key in list(couplings):
            
            if type(key) == str:
                
                couplings[self.TLS_labels[key]] = couplings[key]
                
                del couplings[key]
                
                
        
            
        # make new TLS instance:
        
        new_TLS = TwoLevelSystem(self, TLS_label, is_qubit, energy, couplings, Ls)        
            
        
        # add label to reference dictionary if one is specified
        
        if TLS_label != '':
            
            self.TLS_labels[TLS_label] = new_TLS    
            
            
        
        # save TLS instance to container:
            
        self.TLSs.append(new_TLS)
        
            
        
        
    # builds all full system operators -- currently to call after any changes:    
    
    def build_operators(self):
        
        self.build_Ls()
        
        self.build_H()
        
        self.build_initial_DM() # note: sometimes not required but included for simplicity as cost fairly low
    
        
    
    
    # builds full system Lindblad operators (dissipators)
    
    def build_Ls(self):
        
        
        
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
        
        
        
    
    # builds full system Hamiltonian:
    
    def build_H(self):

        # note: first term in Hamiltonian required at start as qutip can only add like objects
    
    
        # make energy term for first TLS:
            
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
            
            
            
        # add coupling terms for each TLS that has any coupling:
        # note: for now coupling is only recorded in one of each coupled pair -- later to change
        
        for TLS in self.TLSs:
            
            
            # go over all partners to this TLS:
            
            for partner in TLS.couplings:
                
                
                # go over all couplings to this partner:
                # note: this may be a single tuple or a list of them (in case of multiple types of coupling to this partner)
                # hence type check here and convert to list to allow loop
                
                all_couplings_properties = TLS.couplings[partner]
                
                if type(all_couplings_properties) != list:
                    
                    all_couplings_properties = [all_couplings_properties]
                    
                for single_coupling_properties in all_couplings_properties:
               
               
                   # add Hamiltonian term for this partner and this type of coupling:
                       
                
                    strength, op_on_TLS, op_on_partner = single_coupling_properties
                    
                    
                    temp = 1
                    
                    # go over all places, putting correct operators in right places and identities elsewhere
                    
                    for place in self.TLSs:
                        
                        if place == TLS:
                            
                            temp = T(temp, ops[op_on_TLS]*strength) # rescaled by strength here - do not rescale at partner!
                            
                        elif place == partner:
                            
                            temp = T(temp, ops[op_on_partner])
                            
                        else:
                            
                            temp = T(temp, ops['id2'])
                
                    H = H + temp
            
        self.H = H
        
        return H
    
    
    
    
    # builds default initial state density matrix:
    # assumes all defects are in ground state and all qubits excited
    
    def build_initial_DM(self):
        
        temp = 1
        
        for TLS in self.TLSs:
            
            if TLS.is_qubit:
                
                temp = T(temp, (ops['exc'] + ops['sigmay']).unit())
                
            else:
                
                temp = T(temp, ops['gnd'])
                
        self.initial_DM = temp
        
        
    
    
    # calculates dynamics and returns observable measurements at evaluation_times,
    # 1st qubit excited population used unless otherwise specified
    # observable op: if not specified or False, assume by default population in site basis of first qubit in system
    # ...if string and matching existing operator, take that observable on the first qubit with identities elsewhere
    
    def calculate_dynamics(self, 
                          evaluation_times,
                          observable_ops = False,
                          # default: qutip -- unless liouvillian==True
                          dynamics_method = False
                          ):
        
        
        
        # check method argument:
            
        if dynamics_method == False: dynamics_method == 'qutip'
        
        if dynamics_method not in ['qutip', 'liouvillian']:
            
            # print('dynamics calculation method not recognised, hence using qutip')
            
            dynamics_method = 'qutip'
        
        
        
        # check H and initial state available:
            
        if self.H == None:
            
            raise RuntimeError('Hamiltonian not defined -- cannot calculate dynamics')
            
        if self.initial_DM == None:
            
            raise RuntimeError('initial not defined -- cannot calculate dynamics')
            
            
        
        # for observables on single qubit: (either if not specified, a single string, or a list of strings)    
        # make list of total system observable operators:
        # if single TLS operator defined: apply on first qubit in TLSs and identity on others
        
        # not defined hence assume excited population:
        if type(observable_ops) == bool and not observable_ops: observable_ops = 'exc'
        
        # if single operator label (string):
        if isinstance(observable_ops, str): observable_ops = [observable_ops]
        
        
        # tiggered if not valid input valid (should be list with zero non-string elements):
        if not (isinstance(observable_ops, list) 
                and sum([True for x in observable_ops if type(x) != str]) == 0):
            
            raise RuntimeError('do not understand oservable operators input for dynamics calculation')
            
        else: # now have a list of strings for different observable operators and so far assuming they match an operator
        
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
                
                
                
        # solve ME and return observable data array:
          
        switch_print_profiling = False  
            
        
        
        # solve ODE using qutip (default option):
        # return list of arrays of observables values across evaluation times, in order of observable operators
                
        if dynamics_method == 'qutip':
            
            clock = time.time()

            qutip_dynamics = qutip.mesolve(self.H,
                                     self.initial_DM,
                                     evaluation_times,
                                     c_ops = self.Ls,
                                     e_ops = observable_ops_full,
                                     options = qutip.Options(nsteps = 1e9) # store_states = True
                                     # note: creates instance of Options which has as variables all the solver options - can be set in constructor
                                     )
            
            qutip_observables = qutip_dynamics.expect # CONTINUE HERE - CHANGE TO GET ALL THE ARRAYS FOR ALL THE OBSERVABLES!!

            if switch_print_profiling: ('Using qutip:\n' + str(time.time() - clock)) 
            
            return qutip_observables # qutip_dynamics.states
    
    
        
        # build Liouvillian and exponentiate:
        
        elif dynamics_method == 'liouvillian':    
            
            clock = time.time()
            
                
            # H and list of Ls as numpy arrays:
        
            Ls = [np.array(L) for L in self.Ls]
        
            H = np.array(self.H)
            
            
            # vectorised initial DM as numpy array:
    
            initial_DM_mat = np.array(self.initial_DM)            
    
            n = len(initial_DM_mat)
            
            DM_vect = np.reshape(initial_DM_mat, (int(n**2), 1))
            
            
            # Liouvillian:
                
            I = np.eye(n)
                
            LLN = -1j*(np.kron(I, H) - np.kron(H.transpose(), I))
            
            for L in Ls:
                
                LLN += (np.kron(L.conjugate(), L) - 1/2*(np.kron(I, L.conjugate().transpose()@L) + np.kron(L.transpose()@L, I)))
            
            self.LLN = LLN # save to instance for testing
            
            
            # propagator by time interval:
            # note: assumes evaluation_times evenly spaced!
            
            dt = evaluation_times[1] - evaluation_times[0]
            
            clock2 = time.time()
            
            # note: diagonalising and exponentiating: cost of diagonalisation comparable to exponentiation
            # also there must be a precision problem -- Liouvillian should be diagonalisable but transforming
            # diagonal matrix of eigenvalues back gives a slightly different matrix and dynamics is divergent

            P = (sp.linalg.expm(dt*LLN))
            
            # sparse_P = sp.sparse.linalg.expm(sparse_dt_LLN) # sparse form of propagator
            
            time_expm = time.time() - clock2
            
            self.P = P # save to instance for testing
            
            
            # propagate and obtain observables at argument times:
                
            liouvillian_observable = []
            
            time_prop = 0
                
            for i in range(len(evaluation_times)):
            
                if i > 0: DM_vect = P@DM_vect # evolve one time step
                
                DM_mat = np.reshape(DM_vect, (n, n)) # reshape new DM into matrix
                
                clock2 = time.time()
                                
                new_value = np.trace(np.array(observable_op)@DM_mat)

                time_prop += time.time() - clock2
                
                liouvillian_observable.append(new_value) # extract and save observable
                
            if switch_print_profiling:    
                print('__________\nUsing liouvillian:\n' + str(time.time() - clock)) 
                print('exponentiation:\n' + str(time_expm))
                print('propagation total:\n' + str(time_prop))
            
            return liouvillian_observable    
                
            
            
        
        
    # display full model parameters: 
    
    def disp(self):
        
        print(self.model_description_str())
                    
            
    
    
    # return model description as a string (to either print or save to file):
    
    def model_description_str(self, concise = True):
        
        
        # make big string:
        outstring = '______Model:______'    
                
        for TLS in self.TLSs:
            
            
            if TLS.is_qubit: outstring += ('\n\n(' + str(self.TLSs.index(TLS)) + ') - Qubit')
            
            else: outstring += ('\n\n(' + str(self.TLSs.index(TLS)) + ') - Defect')
            
            
            outstring += ('\nEnergy: ' + str(TLS.energy))
            
            outstring += ('\nCouplings:')
            
            
            if concise: # concise version

                for partner in TLS.couplings: # note: each iterand: partner TLS
                    
                    outstring += ('\n   Partner: TLS (' + str(self.TLSs.index(partner)) + ')')
                    
                    for coupling in TLS.couplings[partner]: # note: each iterand: tuple of individual coupling term details 
                        
                        outstring += ('\n      (' + str(coupling[0]) + ', ' + (coupling[1]) + ', ' + (coupling[2]) + ')')
                        
                outstring += ('\nLindblad processes:')
                
                for L in TLS.Ls: # note: each iterand: Lindblad process        
                    
                    outstring += ('\n   ' + L + ': ' + str(TLS.Ls[L]))
                    
                    
            else: # verbose version
                
                for partner in TLS.couplings: # note: each iterand: partner TLS
                    
                    outstring += ('\n   Partner: TLS ' + str(self.TLSs.index(partner)))
                    
                    for coupling in TLS.couplings[partner]: # note: each iterand: tuple of individual coupling term details 
                        
                        outstring += (' \n      Strength: ' + str(coupling[0]))
                        outstring += (' \n      On this: ' + (coupling[1]))
                        outstring += (' \n      On partner: ' + (coupling[2]))
                        
                outstring += (' \nLindblad processes:')
                
                for L in TLS.Ls: # note: each iterand: Lindblad process        
                    
                    outstring += (' \n   Type: ' + L + ' Rate: ' + str(TLS.Ls[L]))
                    
            
        outstring += '\n__________________'
            
        return outstring
    
    
    
    
    # return model description as a JSON compatible dictionary:
    
    def model_description_dict(self):
        
        
        # make nested dictionary:
            
        outdict = {}    
                
        for TLS in self.TLSs:
            
            
            subdict = {}
            
            TLS_id = str(self.TLSs.index(TLS))
            
            if TLS.is_qubit: subdict['type'] = 'Qubit'
            
            else: subdict['type'] = 'Defect'
            
            subdict['energy'] = TLS.energy
            
            
            subsubdict = {}
            
            for partner in TLS.couplings: # note: each iterand: partner TLS
                
                subsubdict[str(self.TLSs.index(partner))] = (TLS.couplings[partner])
                
                # for coupling in TLS.couplings[partner]: # note: each iterand: tuple of individual coupling term details 
                    
                #     outstring += ('\n      (' + str(coupling[0]) + ', ' + (coupling[1]) + ', ' + (coupling[2]) + ')')
                    
            subdict['couplings:'] = subsubdict


            subsubdict = {}
            
            for L in TLS.Ls: # note: each iterand: Lindblad process        
                
                subsubdict[L] = TLS.Ls[L]
                
            subdict['Ls:'] = subsubdict
            
            
            outdict[TLS_id] = subdict
                
                    
        return outdict
    
    
    
    
    # randomly populate model: (used for initial conditions)
    # ! calling this replaces any previous configuration
    # now assuming 1 qubit and couplings only between it and single defects
    
    def random_setup(self,
                     qubits_number = 1, # now assumed one
                     defects_number_range = (2,2), # range given by two-element tuple
                     qubits_energies = [5], # list of floats the length of qubits_number - assumed known
                     defects_energies_range = (3, 7), # tuple of two floats for range
                     allowed_couplings = { # identifier: now should be qubit due to restrictions
                                         'qubit': [(1, (0.1, 1) , 'sigmax', 'sigmax')] # partner: [(probability, rate range, type)]
                                                   #...list for different couplings to this partner
                                         },
                     allowed_Ls = {'sigmaz': (1, (0.005, 0.1)) # type: (probability, rate range)
                                         }
                     ):
    
    
        
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
            
            for partner, val in allowed_couplings.items(): # now redundant since only assuming coupling to qubit
            
            # for this partner now go over all possible couplings:
                
                complete = []
                
                for coupling in val:
            
                    # decide if included based on probability:        
            
                    if not np.random.uniform() < coupling[0]: # ie. NOT included 
                    
                        continue
                    
                    else: # ie. included
                        
                        complete.append((np.random.uniform(*coupling[1]), coupling[2], coupling[3]))
                        
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
        
        
        
        
    
            