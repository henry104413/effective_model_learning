"""

@author: henry
"""


from definitions import T, ops

from TLS import TLS

from qutip import mesolve as qutip_mesolve

from qutip import Options as qutip_Options




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
        
        new_TLS = TLS(self, TLS_label, is_qubit, energy, couplings, Ls)        
            
        
        # add label to reference dictionary if one is specified
        
        if TLS_label != '':
            
            self.TLS_labels[TLS_label] = new_TLS    
            
            
        
        # save TLS instance to container:
            
        self.TLSs.append(new_TLS)
        
            
        
        
    # builds all full system operators -- currently to call after any changes:    
    
    def build_operators(self):
        
        self.build_Ls()
        
        self.build_H()
        
        self.build_initial_DM()
    
        
    
    
    # builds full system Lindblad operators (dissipators)
    
    def build_Ls(self):
        
        
        # total number of subsystems (includes defects and qubits):
            
        n = len(self.TLSs)
        
        
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
                
                    temp = ops[L]
                    
                else: # else start with identity
                    
                    temp = ops['identity']
                
            
                # put subsequent identities or this L in correct places: 
            
                for place in self.TLSs[1:]:
                    
                    if place == TLS:
                        
                        temp = T(temp, ops[L])
                        
                    else:
                        
                        temp = T(temp, ops['identity'])
                                 
                
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
            
            H = T(H, ops['identity'])
            
            
            
        # add energy terms for remaining TLSs:
            
        for TLS in self.TLSs[1:]:
        
            
            # build term for this TLS with energy-setting in correct place and identities elsewhere:
            
            temp = ops['identity']
        
            for place in self.TLSs[1:]:
            
                if place == TLS:
            
                    temp = T(temp, ops['sigmaz']*TLS.energy)
            
                else:    
                
                    temp = T(temp, ops['identity'])
                    
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
                    
                    
                    print(all_couplings_properties)
                    
                
                for single_coupling_properties in all_couplings_properties:
               
               
                   # add Hamiltonian term for this partner and this type of coupling:
                       
                    print(single_coupling_properties)
                    
                    print(type(single_coupling_properties))
               
                    strength, op_on_TLS, op_on_partner = single_coupling_properties
                    
                    
                    temp = 1
                    
                    # go over all places, putting correct operators in right places and identities elsewhere
                    
                    for place in self.TLSs:
                        
                        if place == TLS:
                            
                            temp = T(temp, ops[op_on_TLS]*strength) # rescaled by strength here - do not rescale at partner!
                            
                        elif place == partner:
                            
                            temp = T(temp, ops[op_on_partner])
                            
                        else:
                            
                            temp = T(temp, ops['identity'])
                
                    H = H + temp
            
        self.H = H
        
        return H
    
    
    
    
    # builds default initial state density matrix:
    # assumes all defects are in ground state and all qubits excited
    
    def build_initial_DM(self):
        
        temp = 1
        
        for TLS in self.TLSs:
            
            if TLS.is_qubit:
                
                temp = T(temp, ops['exc'])
                
            else:
                
                temp = T(temp, ops['gnd'])
                
        self.initial_DM = temp
        
        
    
    
    def calculate_dynamics(self, 
                          evaluation_times,
                          observable_op = False):
        
        
        
        # check H and initial state available:
            
        if self.H == None:
            
            raise RuntimeError('Hamiltonian not defined -- cannot calculate dynamics')
            
        if self.initial_DM == None:
            
            raise RuntimeError('initial not defined -- cannot calculate dynamics')
            
            
            
        # default observable operator unless another specified:
        # now using excited population of first qubit in TLSs
        
        if type(observable_op) == bool and not observable_op:
        
            temp = 1
            
            already_got_one = False
        
            for TLS in self.TLSs:
                
                if TLS.is_qubit and not already_got_one:
                    
                    temp = T(temp, ops['exc'])
                    
                else:
                    
                    temp = T(temp, ops['identity'])
        
            observable_op = temp
            
            
            
        # solve ME and return observable data array:
        
        dynamics = qutip_mesolve(self.H,
                                 self.initial_DM,
                                 evaluation_times,
                                 c_ops = self.Ls,
                                 e_ops = [observable_op],
                                 options = qutip_Options(nsteps = 1e9)
                                 )
        
        return observable.expect[-1]

        