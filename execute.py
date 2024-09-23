"""

@author: henry
"""

from model import Model

from learning_model import LearningModel

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

ts = np.linspace(0, 5e1, int(1000))#int(1000))

measurements = GT.calculate_dynamics(ts)



# initial guess:

initial_guess = LearningModel()


initial_guess.add_TLS(TLS_label = 'qubit',
                     is_qubit = True,
                     energy = 5,
                     couplings = {
                                  
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )


initial_guess.add_TLS(is_qubit = False,
                     energy = 4.0,
                     couplings = {'qubit': [(0.4, 'sigmax', 'sigmax')]
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )

initial_guess.build_operators()



# learning:


# create instance of learning (quest for best model):
quest = LearningChain(target_times = ts, target_data = measurements, initial_guess = initial_guess)

costs = quest.learn(5)

best_data = quest.best.calculate_dynamics(ts)


#%%

# test plot:





plt.figure()
plt.plot(ts*Constants.t_to_sec, measurements, 'r-', label = 'ground truth')
plt.plot(ts*Constants.t_to_sec, best_data, 'b--', label = 'learned model')
plt.xlabel('time (fs)')
plt.ylabel('qubit excited population')
plt.ylim([0,1.1])


plt.figure()
#plt.plot(list(range(len(costs))),costs)
plt.plot(costs[0:10])
plt.xlabel('iteration')
plt.ylabel('cost')