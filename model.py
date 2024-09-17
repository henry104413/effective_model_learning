"""

@author: henry
"""


from definitions import T, ops, tensor_product_starter

from TLS import TLS

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
        
            
    
    def build_operators(self):
        
        self.build_Ls()
    
        
        pass
    
    
    
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
        
        for TLS
            
            
        self.H = H
        
        return H
    
    
    
    
    def build_initial_DM():
        
        pass
    
    
    def evaluate_dynamics(evaluation_times,
                          observable_operator = False):
        
        
        pass