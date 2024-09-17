"""

@author: henry
"""

from model import Model

from learning_model import LearningModel

from definitions import ops, T


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
                     energy = 4,
                     couplings = {'qubit': (0.5, 'sigmap', 'sigmam')
                                  
                                  },
                     Ls = {
                           'sigmaz' : 0.01
                           }
                     )