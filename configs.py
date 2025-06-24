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

couplings_shape_scale = (0.8, 1)
Ls_shape_scale = (0.2, 0.5)

default_chain_hyperparams = {    
        'chain_step_options': {
            'tweak all parameters': 5,
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
                                     'energies' : 0.9,
                                     'Ls' : 0.010
                                     }
            },
        
        'qubit_Ls_library': { # sampled from mirrored gamma distribution with given (shape, scale)
           'sigmax': Ls_shape_scale#(0.01, 0.1)
           ,'sigmay': Ls_shape_scale#(0.01, 0.1)
           ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
           },
     
        'defect_Ls_library': { # sampled from mirrored gamma distribution with given (shape, scale)
           'sigmax': Ls_shape_scale#(0.01, 0.1)
           ,'sigmay': Ls_shape_scale#(0.01, 0.1)
           ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
           },
     
        'qubit2defect_couplings_library': { # sampled from mirrored gamma distribution with given (shape, scale)
           (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
           },
        
        'defect2defect_couplings_library': { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },
        
        'params_thresholds': { # minimum values for parameters - if below then process dropped
            # !!! does this break reversibility??                
            'Ls':  1e-7,
            'couplings': 1e-6
            },
        
        'custom_function_on_dynamics_return': False, #custom_func
        
        'iterations_till_progress_update': 100
}        
        


specific_experiment_chain_hyperparams = {
  
'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         #'sigmax': Ls_shape_scale#(0.01, 0.1)
         #,'sigmay': Ls_shape_scale#(0.01, 0.1)
         'sigmaz': Ls_shape_scale#(0.01, 0.1)
         },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         #'sigmax': Ls_shape_scale#(0.01, 0.1)
         #,'sigmay': Ls_shape_scale#(0.01, 0.1)
         #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
         },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
         #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
         #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
         },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
         #(('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
         #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
         #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
         }
    },
    
'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v-sx-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v-sx-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx-Cv2v-sx-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #'sigmax': Ls_shape_scale#(0.01, 0.1)
          #,'sigmay': Ls_shape_scale#(0.01, 0.1)
          #,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx,sy,sz-Cv2v--':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          #(('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          #,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    },
    
'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
    {
    'qubit_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'defect_Ls_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          'sigmax': Ls_shape_scale#(0.01, 0.1)
          ,'sigmay': Ls_shape_scale#(0.01, 0.1)
          ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
          },

    'qubit2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          },

    'defect2defect_couplings_library':
        { # sampled from mirrored gamma distribution with given (shape, scale)
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
          }
    }
   
}
