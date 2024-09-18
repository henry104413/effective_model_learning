"""

@author: henry
"""

from model import Model

# from learning_model import LearningModel

# from definitions import ops, T

from definitions import Constants

import numpy as np

import matplotlib.pyplot as plt



ground_truth = Model()


ground_truth.add_TLS(TLS_label = 'qubit',
                     is_qubit = True,
                     energy = 5,
                     couplings = {
                                  
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )


ground_truth.add_TLS(is_qubit = False,
                     energy = 4.5,
                     couplings = {'qubit': [(0.5, 'sigmap', 'sigmam'), (0.5, 'sigmam', 'sigmap')]
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )
    
    
ground_truth.build_operators()



#%%
GT_times = np.linspace(0, 1e1, int(1000))

GT_qubit_evo = ground_truth.calculate_dynamics(GT_times)


plt.figure()
plt.plot(GT_times*Constants.t_to_sec, GT_qubit_evo, 'r-')
plt.xlabel('time (fs)')
plt.ylabel('qubit excited population')
plt.ylim([0,1.1])