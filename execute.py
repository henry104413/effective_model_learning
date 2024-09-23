"""

@author: henry
"""

from model import Model

# from learning_model import LearningModel

from learning_chain import LearningChain

from definitions import Constants

import numpy as np

import matplotlib.pyplot as plt



# set up ground truth model:

GT = Model()

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
                     couplings = {'qubit': [(0.5, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )
        
GT.build_operators()


# generate data measurements:
# note: now using 1st qubit excited population at times ts

ts = np.linspace(0, 3e1, int(1000))

measurements = GT.calculate_dynamics(ts)



# create instance of learning (model search):
search = LearningChain(target_times = ts, target_data = measurements)

#search.run()


#%%

# test plot:





plt.figure()
plt.plot(ts*Constants.t_to_sec, measurements, 'r-', label = 'ground truth')
#plt.plot(ts*Constants.t_to_sec, search.best_data(), 'r-')
plt.xlabel('time (fs)')
plt.ylabel('qubit excited population')
plt.ylim([0,1.1])