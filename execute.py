"""

@author: henry
"""

from model import Model

from learning_model import LearningModel

# from learning_model import LearningModel

# from definitions import ops, T

from definitions import Constants

import numpy as np

import matplotlib.pyplot as plt



# set up ground truth model:

GT = LearningModel()


GT.add_TLS(TLS_label = 'qubit',
                     is_qubit = True,
                     energy = 5,
                     couplings = {
                                  
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )


GT.add_TLS(is_qubit = False,
                     energy = 4.5,
                     couplings = {'qubit': [(0.5, 'sigmap', 'sigmam'), (0.5, 'sigmam', 'sigmap')]
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )
    
    
GT.build_operators()



guess = LearningModel()


# ADD MODEL PRINTING METHOD!


#%%

# test plot:

GT_times = np.linspace(0, 3e1, int(1000))

GT_qubit_evo = GT.calculate_dynamics(GT_times)


plt.figure()
plt.plot(GT_times*Constants.t_to_sec, GT_qubit_evo, 'r-')
plt.xlabel('time (fs)')
plt.ylabel('qubit excited population')
plt.ylim([0,1.1])