"""

Global definitions and shorthands used throughout the Effective Model Learning codebase.

@author: henry

"""

import qutip as q
import numpy as np




#%% global definitions and shorthands:



# constants:
# note: h_bar=1, e=1
class Constants():
    
    # conversion factors:
    K_to_eV = 1.381/1.602*1e-4 # multiply by T in K to get energy (k_B*T) in eV
    t_to_sec = 4.136e-15 # multiply by time values before plotting to get the value in seconds


def T(arg1: q.Qobj, arg2: q.Qobj) -> q.Qobj:
    """
    Returns tensor product of arguments as computed by qutip.
    
    Also works when one factor being a number, just returning other factor.
    Note: This is useful as starting point for loops gradually constructing products.
    """
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
       'plus': q.ket2dm((q.ket([1]) + q.ket([0])).unit()), # careful - qutip uses possibly opposite convention for excited and ground state
       '2e+1g': q.ket2dm((2*q.ket([1]) + 1*q.ket([0])).unit()),
       'custom': q.Qobj([[0.5,0.5],[0.5,0.5]]),
       'mm': q.maximally_mixed_dm(2)
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

observable_shorthand2pretty = {'sx': r'$\langle\sigma_x\rangle$',
                               'sy': r'$\langle\sigma_y\rangle$',
                               'sz': r'$\langle\sigma_z\rangle$',
                               'sp': r'$\langle\sigma_+\rangle$',
                               'sm': r'$\langle\sigma_-\rangle$',
                               'sigmax': r'$\langle\sigma_x\rangle$',
                               'sigmay': r'$\langle\sigma_y\rangle$',
                               'sigmaz': r'$\langle\sigma_z\rangle$',
                               'sigmap': r'$\langle\sigma_+\rangle$',
                               'sigmam': r'$\langle\sigma_-\rangle$'
                               }


# currently unused: 
# rate bounds:
rates_bounds = {'sigmaz' : (0.001, 0.1),
                'sigmax' : (0.001, 0.1),
                'sigmay' : (0.001, 0.1),
                'sigmam' : (0.001, 0.1),
                'sigmap' : (0.001, 0.1),
                }
rates_arrays = {}
for key in rates_arrays:
    rates_arrays[key] = np.logspace(*rates_bounds[key], 10)

    
# currently unused:
def MSE(A, B):
    """
    Returns mean squared error of two argument numpy arrays.
    """
    
    # check type: (must be numpy arrays to use operators below)
    if type(A) != type(np.array([])) or type(B) != type(np.array([])):
        raise RuntimeError('error calculating MSE: arguments must be numpy arrays!\n')
    
    # check matching length:
    if len(A) != len(B):
        raise RuntimeError('error calculating MSE: arguments must be same length!\n')
        
    # calculate mean squared error:    
    return np.sum(np.square(np.abs(A-B)))/len(A)


#%%
