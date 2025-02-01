#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import copy
import numpy as np


 
#%%



# executes any modification to model processes (operators)
# stores available process library

# methods should take model as argument and work on it
# same process handling object expected to be used on various models 

class ProcessHandler():
    
    
    
    def __init__(self, chain = False, model = False,
                 Ls_library = False,
                 qubit_couplings_library = False,
                 defect_couplings_library = False):

        self.Ls_library = Ls_library   # dictionary: {'op label': (lower_bound, upper_bound)}
        self.initial_Ls_library = copy.deepcopy(Ls_library)
        self.qubit_couplings_library = qubit_couplings_library # also {'op label': (lower_bound, upper_bound)}
        self.defect_couplings_library = defect_couplings_library # also {'op label': (lower_bound, upper_bound)}
        self.initial_qubit_couplings_library = copy.deepcopy(qubit_couplings_library)
        self.model = model # only for future methods tied to specific model
        self.chain = chain # only for future methods tied to chain (eg using its cost function)

        

    # adds random new single-site Linblad process from process library to random subsystem (qubit or defect for now) 
    # with rate drawn from uniform distribution between bounds given by tuple for each process library entry  
    # works on and returns argument model
    # argument process library used if passed, otherwise use that of handler if already set, error if neither available  
    def add_random_L(self, model, Ls_library = False):
        
        # check library available:
        if not Ls_library:
            if isinstance(self.Ls_library, dict):
                Ls_library = self.Ls_library
            else:
                raise RuntimeError('Cannot add Lindblad process as process library not specified')
            
        # select subsystem:
        subsystems = [x for x in model.TLSs]# if not x.is_qubit]
        subsystem = np.random.choice(subsystems)
        
        # identify Ls in library not yet present on selected subsystem:
        existing = [x for x in subsystem.Ls]
        options = [x for x in Ls_library if x not in existing] # note: could just put subsystem.Ls instead of existing...
        
        # select new operator if available, draw rate from uniform distribution between bounds, and add:
        if options:
            operator = np.random.choice(options)
            new_rate = np.random.uniform(*Ls_library[operator])
            subsystem.Ls[operator] = new_rate 
            model.build_operators()
            # print('adding operator ' + operator + ' to TLS no. ' + str(model.TLSs.index(subsystem)) + ' with rate ' + str(new_rate))
            # note: this directly modifies variable another classe's object
            # ...perhaps could be changed to instead work via method of that class?
            
        else:
            #print('no options for adding operator to TLS no. ' + str(model.TLSs.index(subsystem)))
            pass
        
        return model
    
    
    
    # adds random single-operator coupling between first qubit and random defect:
    # note: this way the single operator should be hermitian and same on both subsystems
    # future: expand to multiple operators - requires rewrtiting coupling generation 
    # note: !! at the moment two-system coupling information is only stored on one of the systems
    # its stored like: { partner : [(strength, op_this, op_partner)]} ...this and partner are TLS objects (pointer)
    # !!!
    def add_random_qubit_coupling(self, model, qubit_couplings_library = False):
        
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
                    #print('coupling via '  + operator + ' already exists')
                    continue
                    
            # likewise check on qubit:
            if defect in [x for x in qubit.couplings]:
                some_coupling = True
                if operator in [y[1] for y in qubit.couplings[defect]]:
                # note: checking this against operator on qubit but both should be same so far until mixed operators implemented
                    #print('coupling via ' + operator + ' already exists')
                    continue
            
            
            # ie coupling via this operator between the two not yet existing:
                
            # draw strenght:
            strength = np.random.uniform(*qubit_couplings_library[operator])
            
            # initialise list if no coupling between the two exists yet:
            if not some_coupling:
                qubit.couplings[defect] = []
            
            # add new coupling:
            #print(qubit.couplings)
            qubit.couplings[defect].append((strength, operator, operator))
            model.build_operators()
            
            return model
        
            break # safety break
            
        
        
    # adds random single-operator coupling between first qubit and random defect:
    # note: this way the single operator should be hermitian and same on both subsystems
    # future: expand to multiple operators - requires rewrtiting coupling generation 
    # note: !! at the moment two-system coupling information is only stored on one of the systems
    # its stored like: { partner : [(strength, op_this, op_partner)]} ...this and partner are TLS objects (pointer)
    # !!!
    def add_random_defect2defect_coupling(self, model, defect_couplings_library = False):
        
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
                    #print('coupling via '  + operator + ' already exists')
                    continue
                    
            # likewise check on defect2:
            if defect1 in [x for x in defect2.couplings]:
                
                if operator in [y[1] for y in defect2.couplings[defect1]]:
                # note: checking this against operator on qubit but both should be same so far until mixed operators implemented
                    #print('coupling via ' + operator + ' already exists')
                    continue
            
            
            # ie coupling via this operator between the two not yet existing:
                
            # draw strenght:
            strength = np.random.uniform(*defect_couplings_library[operator])
            
            # initialise list if no coupling between the two exists yet:
            if not some_coupling:
                defect1.couplings[defect2] = []
            
            # add new coupling:
            #print(qubit.couplings)
            defect1.couplings[defect2].append((strength, operator, operator))
            model.build_operators()
            
            return model
        
            break # safety break
        
         # key error when running again (defect2 as key bugs)
        
    
        
        
        
        
    # removes random Lindblad process from random subsystem
    # works on and returns argument model
    def remove_random_L(self, model):
        
        # select subsystem:
        subsystems = [x for x in model.TLSs]# if not x.is_qubit]
        subsystem = np.random.choice(subsystems)
        
        # identify Ls present on selected subsystem:
        options = [x for x in subsystem.Ls]
        
        # select operator and remove:
        if options:
            operator = np.random.choice(options)
            subsystem.Ls.pop(operator)
            model.build_operators()
            #print('removing operator ' + operator + ' from TLS no. ' + str(model.TLSs.index(subsystem)))
            
        else:
            #print('no options for removing operator from TLS no. ' + str(model.TLSs.index(subsystem)))
            pass
        
        
    
    
    
    # sets process library post-constructor: 
    def define_Ls_library(self, Ls_library):
    
        self.Ls_library = Ls_library
        
        
        
    # resets Ls library to initial one:    
    def reset_Ls_library(self):
        
        self.Ls_library = copy.deepcopy(self.initial_Ls_library)
        
    
    
    # rescales bound distance from middle of range by half of given factor for all process library rates:
    # note: range hence changes by up to factor
    # rates assumed positive and capped at zero from left (hence range can change less)
    def rescale_Ls_library_range(self, factor):
    
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
            
            
            
    # removes random coupling process between qubit and defect:        
    def remove_random_qubit_coupling(self, model):
        
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
            
        
        
    
    # removes random coupling process between two different defects:        
    def remove_random_defect2defect_coupling(self, model):
        
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
            
            