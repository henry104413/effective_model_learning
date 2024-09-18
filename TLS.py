"""

@author: henry
"""

class Two_level_system():
    
    
    # makes new TLS instance:
    # arguments: model instance this belongs to; qubit indicator (false means defect); energy;
    # dictionary of coherent couplings where keys are other TLSs in this model; and values are tuples:
    # ...(rate, operator acting on this, operator acting on other TLS);
    # dictionary of linblad operators where key is operator and value is rate
    
    def __init__(self, model, TLS_id = "", is_qubit = False, energy = None, couplings = {}, Ls = {}):
        
        
        # to add: input check
        # inluding: operators defined, and structure of dictionaries... later though
        
        self.model = model
        
        self.TLS_id = TLS_id
             
        self.is_qubit = is_qubit
        
        self.energy = energy
        
        self.couplings = couplings
        
        self.add_couplings(couplings)
        
                 
        # to add: call coupling method to also add coupling to the other system
        # to add: check for duplicates
        # so far: manually ensure they are coupled in the order they are added
        
        
        # this way each TLS contains complete information about it
        # Hamiltonian builder function should then make sure it only applies this once
        # operator builders should probably be model methods as the dimensionality is model dependent
        
        
        self.Ls = Ls
        
        
        
        # every TLS must have an energy!
                
    
    
    # to add coupling information to the other TLS
    # rename this to something more explicit
    def add_couplings(self, couplings):
        
        pass
                 
    
    
                 
                 
                 
                 
                 
                 
                 
                 
                 