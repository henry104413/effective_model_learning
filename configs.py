#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 16:29:31 2025

@author: henry
"""

# shorthands for hyperparams definitions:
couplings_shape_scale = (0.8, 1)
Ls_shape_scale = (0.2, 0.5)


configs = {
  
'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v--':
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
    },
    
'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v-sx-':
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
    
'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--':
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
    },
    
'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-':
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
          (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
          ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
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