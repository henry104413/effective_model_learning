"""

Global definitions used throughout the Effective Model Learning codebase.

@author: henry

"""

import qutip as q
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as tkr
from copy import deepcopy
from scipy.optimize import minimize
from scipy import interpolate




#%% global definitions:



# constants:

class Constants():
    
        
    # note: h_bar=1, e=1
    
    
    # conversion factors:
    
    K_to_eV = 1.381/1.602*1e-4 # multiply by T in K to get energy (k_B*T) in eV
    
    t_to_sec = 4.136e-15 # multiply by time values before plotting to get the value in seconds




# now how about something really clever...
# because NOW I should be able to start tensor products with just temp = 1

def T(arg1, arg2):
    
    if type(arg1) == type(q.Qobj()):
        
        return q.tensor(arg1, arg2)
    
    elif type(arg1) in [int, float]: # usually passed as 1
    
        return arg2



# 2-D operators definition using dictionary keys:

ops = {#'sigma z' : q.sigmaz(),
       'sigmaz' : q.sigmaz(),
       #sigma x' : q.sigmax(),
       'sigmax' : q.sigmax(),
       #'sigma y' : q.sigmay(),
       'sigmay' : q.sigmay(),
       #'sigma plus' : q.sigmap(),
       'sigmap' : q.sigmap(),
       #'sigma minus' : q.sigmam(),
       'sigmam' : q.sigmam(),
       #'identity' : q.identity(2),
       #'id' : q.identity(2),
       'id2' : q.identity(2),
       'gnd' : q.sigmam()*q.sigmap(),
       'exc' : q.sigmap()*q.sigmam(),
       'plus': q.ket2dm(q.ket([1]) + q.ket([0])).unit() # careful - qutip uses possibly opposite convention for excited and ground state
       }



# operator parameter labels for plotting:
    
ops_labels = {'defects energies' : 'energy (eV)',
              'defects couplings' : 'coupling (eV)',
              'sigma z' : r'$\sigma_z$' + ' rate (eV)',
              'sigma x' : r'$\sigma_x$' + ' rate (eV)',
              'sigma y' : r'$\sigma_y$' + ' rate (eV)',
              'sigma plus' : r'$\sigma_+$' + ' rate (eV)',
              'sigma minus' : r'$\sigma_-$' + ' rate (eV)'
              }



rates_bounds = {'sigmaz' : (0.001, 0.1),
                'sigmax' : (0.001, 0.1),
                'sigmay' : (0.001, 0.1),
                'sigmam' : (0.001, 0.1),
                'sigmap' : (0.001, 0.1),
                }


rates_arrays = {}

for key in rates_arrays:
    
    rates_arrays[key] = np.logspace(*rates_bounds[key], 10)


    

# calculates mean squared error of two arrays:
# arguments: (numpy array, numpy array)


def MSE(A, B):
    
    
    # check type: (must be numpy arrays to use operators below)
    
    if type(A) != type(np.array([])) or type(B) != type(np.array([])):
    
        raise RuntimeError('error calculating MSE: arguments must be numpy arrays!\n')
        
    
    # check matching length:
    
    if len(A) != len(B):
        
        raise RuntimeError('error calculating MSE: arguments must be same length!\n')
        
        
    # calculate mean squared error:    
    
    return np.sum(np.square(np.abs(A-B)))/len(A)


#%%
