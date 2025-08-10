#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Chain instance hyperparameter configuration file.


Usage: Dictionary where keys are names of LearningChain initialiser arguments.

default_chain_hyperparams contains entries for default configuration superseding LearningChain defaults class.

experiment_specific_chain_hyperparams is a dictionary where keys are different experiment names,
and entris are hyperparams dictionaries that overwrite (supersede) given defaults dictionary entries.


@author: Henry (henry104413)
"""
import copy

couplings_shape_scale = (2, 0.3)
Ls_shape_scale = (2, 0.03)

default_chain_hyperparams = {    
        'chain_step_options': {
            'tweak all parameters': 32,
            'add qubit L': 1,
            'remove qubit L': 1,
            'add defect L': 1,
            'remove defect L': 1,
            'add qubit-defect coupling': 1, 
            'remove qubit-defect coupling': 1,
            'add defect-defect coupling': 1, 
            'remove defect-defect coupling': 1
            },
        
        'temperature_proposal': 0.0005, # either value or (shape, scale) of gamma to sample
        
        'jump_length_rescaling_factor': 1.0, # for scaling up or down jump lengths of parameter handler
        
        'acceptance_window': 10,
        'acceptance_target': 0.4,
        'acceptance_band': 0.2,
        
        'params_handler_hyperparams': { 
            'initial_jump_lengths': {'couplings' : 0.10,
                                     'energies' : 0.10,
                                     'Ls' : 0.010
                                     }
            },
        
        'qubit_Ls_library': { # sampled from mirrored gamma distribution with given (shape, scale)
           'sigmax': Ls_shape_scale
           ,'sigmay': Ls_shape_scale
           ,'sigmaz': Ls_shape_scale
           },
     
        'defect_Ls_library': { # sampled from mirrored gamma distribution with given (shape, scale)
           'sigmax': Ls_shape_scale
           ,'sigmay': Ls_shape_scale
           ,'sigmaz': Ls_shape_scale
           },
     
        'qubit2defect_couplings_library': { # sampled from mirrored gamma distribution with given (shape, scale)
           (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
           },
        
        'defect2defect_couplings_library': { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },
        
        'params_thresholds': { # minimum values for parameters - if below then process dropped
            # currently discontinued - sampling from distributions not heavy around zero seems to dispense with need
            # also priors introduced decaying at zero and +inf                
            'Ls':  1e-7,
            'couplings': 1e-6
            },
        
        'params_priors': { # (shape, scale) for gamma dristributions each for one parameter class
            'couplings': (1.04, 30),
            'energies': (1.05, 35),   
            'Ls': (1.004, 23)
            },
        
        'custom_function_on_dynamics_return': False, #custom_func
        
        'iterations_till_progress_update': 100
}        
        


specific_experiment_chain_hyperparams = {
# 0)  
'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         #'sigmax': Ls_shape_scale
         #,'sigmay': Ls_shape_scale
         'sigmaz': Ls_shape_scale
         },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         #'sigmax': Ls_shape_scale
         #,'sigmay': Ls_shape_scale
         #,'sigmaz': Ls_shape_scale
         },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         (('sigmax', 'sigmax'),): couplings_shape_scale
         #,(('sigmay', 'sigmay'),): couplings_shape_scale
         #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
         },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         #(('sigmax', 'sigmax'),): couplings_shape_scale
         #,(('sigmay', 'sigmay'),): couplings_shape_scale
         #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
         }
    },

# 1)    
'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v-sx-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },
 
 # 2)   
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 3)    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v-sx-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },
 
 # 4)   
'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },
 
 # 5)   
'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx-Cv2v-sx-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },
 
 # 6)   
'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 7)    
'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 8)    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 9)    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 10)    
'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx,sy,sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },
 
 # 11)   EVERYTHING
'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          ,'sigmay': Ls_shape_scale
          ,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 12)    
'Lsyst-sx-Lvirt--Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 13)    
'Lsyst-sx-Lvirt--Cs2v-sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          (('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 14)    
'Lsyst-sz-Lvirt--Cs2v-sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          (('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },
    
# 15)
'Lsyst-sz-Lvirt--Cs2v-sx,sy-Cv2v--':
{
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          ,(('sigmay', 'sigmay'),): couplings_shape_scale
          #(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    },

# 16)
'Lsyst-sz-Lvirt--Cs2v-sx,sz-Cv2v--':
{
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          'sigmaz': Ls_shape_scale
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale
          #,'sigmay': Ls_shape_scale
          #,'sigmaz': Ls_shape_scale
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale
          #,(('sigmay', 'sigmay'),): couplings_shape_scale
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale
          }
    }

   
}

    
    
def get_hyperparams(batch_name: str) -> dict:
    """
    Returns dictionary of default hyperparameters
    with inclusion of specific hyperparameters under key given by argument.
    """
    hyperparameters = copy.deepcopy(default_chain_hyperparams)
    hyperparameters.update(specific_experiment_chain_hyperparams[batch_name])
    return hyperparameters
