#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import multiprocessing
import time
import numpy as np

import basic_model
import learning_chain
import output



#%% import target data:
    
# import data from CSV file with possible annotations skipped
# assuming subsequent pairs of columns are different datasets 
# and numberical entries on single row are x, y values

# choose data:
datafile = 'Witnessing_Fig4b.csv'
dataset_no = 0 # starting from 0

# extract x and y values@
contents = np.genfromtxt('Witnessing_Fig4b.csv',delimiter=',')#,dtype=float) 
dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

# sort by x:
order = xs.argsort()
xs = xs[order]
ys = ys[order]

# note: integrator errors - thought cause was unevenly spaced data likely just unsorted
# apparently works if sorted descending ascending or descending, but unsorted breaks!

# times for model evaluation:
ts = xs/1000

# measured data feed:
measurement_datasets = [ys]
measurement_observables = ['sigmax']



#%% perform learning:
    
# if qubit initial state required:
import definitions
qubit_initial_state = definitions.ops['sigmax']

# for trying with absolute value of observable:
def custom_func(arg):
    print('taking abs')
    if isinstance(arg, list): return [abs(x) for x in arg]
    else: return abs(arg)
    
# shorthands for hyperparams definitions:
couplings_shape_scale = (0.8, 1)
Ls_shape_scale = (0.2, 0.5)


# instance of learning (quest for best model):
quest = learning_chain.LearningChain(target_times = ts,
                      target_datasets = measurement_datasets,
                      target_observables = measurement_observables,
                      
                      initial = (1, 2), # (qubit energy, number of defects)
                      qubit_initial_state = qubit_initial_state,
                      
                      max_chain_steps = 100,
                      chain_step_options = {
                          'tweak all parameters': 0.3,
                          'add L': 0.05,
                          'remove L': 0.05,
                          'add qubit-defect coupling': 0.05, 
                          'remove qubit-defect coupling': 0.05,
                          'add defect-defect coupling': 0.05, 
                          'remove defect-defect coupling': 0.05
                          },
                      
                      temperature_proposal = 0.0005, # either value or (shape, scale) of gamma to sample
                      
                      jump_length_rescaling_factor = 1.0, # for scaling up or down jump lengths of parameter handler
                      
                      acceptance_window = 10,
                      acceptance_target = 0.4,
                      acceptance_band = 0.2,
                      
                      params_handler_hyperparams = { 
                          'initial_jump_lengths': {'couplings' : 0.05,
                                                   'energies' : 0.5,
                                                   'Ls' : 0.005
                                                   },
                          },
                      
                      Ls_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         'sigmax': Ls_shape_scale#(0.01, 0.1)
                         ,'sigmay': Ls_shape_scale#(0.01, 0.1)
                         ,'sigmaz': Ls_shape_scale#(0.01, 0.1)
                         },
                   
                      qubit2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
                        ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
                        ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
                         },
                      
                      defect2defect_couplings_library = { # sampled from mirrored gamma distribution with given (shape, scale)
                         (('sigmax', 'sigmax'),): couplings_shape_scale#(0.3, 1)
                        ,(('sigmay', 'sigmay'),): couplings_shape_scale#(0.3, 1)
                        ,(('sigmaz', 'sigmaz'),): couplings_shape_scale#(0.3, 1)
                        },
                      
                      params_thresholds = { # minimum values for parameters - if below then process dropped
                          # !!! does this break reversibility??                
                          'Ls':  1e-7,
                          'couplings': 1e-6
                          },
                      
                      custom_function_on_dynamics_return = False,#custom_func
                      
                      iterations_till_progress_update = False
                      )

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(new_ts := np.linspace(min(ts), max(ts)/1, 1000),
#           quest.initial.calculate_dynamics(new_ts, ['sigmax'])[0])

#%%
best = quest.run()

#%%
best = quest.best

costs = quest.explored_loss
acceptance_ratios = quest.chain_windows_acceptance_log
evaluation_ts = np.linspace(ts[0], ts[-1], max(10*len(ts), int(1000)))
best_datasets = best.calculate_dynamics(evaluation_ts, observable_ops = measurement_observables,
                                        custom_function_on_return = False)


#%% chain run outputs:

# output controls bundle:
class Toggles():    
    comparison = True # plot comparison of dynamics
    loss = True # plot cost function progression
    acceptance = True # plot acceptance ratios over subsequenct windows
    graphs = True # plot model graphs with corresponding labels
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True # save chain hyperparameters as json

# unique name (date and time stamp):
timestamp = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())
# maybe add additional check to in unlikely case parallel run ends up with same name...

# create outputs:
O = output.Output(toggles = Toggles, filename = timestamp,
       dynamics_ts = [ts, evaluation_ts],
       dynamics_datasets = [measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['measured', 'learned'],
       dynamics_formatting = ['b+', 'r-'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       acceptance = acceptance_ratios,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


#%% 
raise SystemExit()


import itertools as it
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


nodes = "ABC"
prod = list(it.product(nodes, repeat=2))
pair_dict = {f"Product x {i}": prod * i for i in range(1, 5)}


fig, axes = plt.subplots(2, 2)
for (name, pairs), ax in zip(pair_dict.items(), np.ravel(axes)):
    G = nx.MultiDiGraph()
    for i, (u, v) in enumerate(pairs):
        G.add_edge(u, v, w=round(i / 3, 2))
    draw_labeled_multigraph(G, "w", ax)
    ax.set_title(name)
fig.tight_layout()
plt.show()
