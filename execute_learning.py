#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""



import numpy as np
import sys # for passing command line arguments
import time
# note: import os also called below in case of an exception


import basic_model
import learning_chain
import output



# run parameters taken from additional command line arguments,
# order: target_file, experiment_name, defects_count, repetition_number, max_iterations;
# defaults specified here if unavailable
# note: files are overwritten if saved with same name

# set target data file:
try:
    target_file = str(sys.argv[1])
except:
    # by default take first csv file found in current folder
    # note: only tested on Linux
    try:
        import os
        target_file = next(x for x in os.listdir() if '.csv' in x)
    except:
        raise SystemExit('Unable to open any csv file - aborting')        

# set experiment name for file naming:
try:
    experiment_name = str(sys.argv[2])
except:
    experiment_name = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())
    
# set number of defects,
try:
	defects_count = int(sys.argv[3])
except: 
	defects_count = 5
	
# set repetition number for file naming:
# note: refers to repetition of run with same defects number
# note: repetitions carried out by external loop (bash script)
try:
	repetition_number = int(sys.argv[4])
except: 
	repetition_number = 1

# set maximum iterations:
try:
    max_iterations = int(sys.argv[5])
    if max_iterations == 0:
        raise Exception('Maximum iterations not specified by launcher, hence using default.')
except:
    max_iterations = 100

# set proportion (ratio) of available data values to use for training:
# note: currently taken from start and same for all data sets; 1 means use all
# note: must be a number and not expression
try:
    proportion_to_use = float(sys.argv[6])
except:
    proportion_to_use = 1

# run's output files common name:
# example: '250421_Wit4b-grey_ForClusters'
filename = (experiment_name + '_D' + str(defects_count) +
    '_R' + str(repetition_number))
 
print(filename, flush = True)


#%% import target data:
    
# import data from CSV file with possible annotations skipped
# assuming subsequent pairs of columns are different datasets 
# and numberical entries on single row are x, y values

# choose data:
datafile = target_file
dataset_no = 0 
# note: 0 means first pair of columns
# note: currently assuming first dataset in file is target now
# and not set by launcher

# extract x and y values@
contents = np.genfromtxt(datafile,delimiter=',')#,dtype=float) 
dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

# sort by x:
order = xs.argsort()
xs = xs[order]
ys = ys[order]

# note: apparently has to be sorted else integrator fails
# (both ascending and descending work, can be unevenly spaced)



#%% data preparation:

# rescaled measurement times:
# note: proper scaling here avoids having to scale other hyperparameters
ts = xs/1000

# full measurement data:
# (encapsulate into lists of datasets and corresponding observable lables)
measurement_datasets = [ys]
measurement_observables = ['sigmax']
    
# times and measurement data to use for training:
# (encapsulate into lists of datasets and corresponding observable lables:)
training_ts = ts[:int(proportion_to_use*len(ts))]
training_ys = ys[:int(proportion_to_use*len(ys))]
training_measurement_datasets = [training_ys]
training_measurement_observables = ['sigmax']



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
quest = learning_chain.LearningChain(target_times = training_ts,
                      target_datasets = training_measurement_datasets,
                      target_observables = training_measurement_observables,
                      
                      initial = (1, defects_count), # (qubit energy, number of defects)
                      qubit_initial_state = qubit_initial_state,
                      
                      max_chain_steps = max_iterations,
                      chain_step_options = {
                          'tweak all parameters': 0.3,
                          'add L': 0.05,
                          'remove L': 0.05,
                          'add qubit-defect coupling': 0.05, 
                          'remove qubit-defect coupling': 0.05,
                          'add defect-defect coupling': 0.05, 
                          'remove defect-defect coupling': 0.05
                          },
                      
                      temperature_proposal = 0.0010, # either value or (shape, scale) of gamma to sample
                      
                      jump_length_rescaling_factor = 1.0, # for scaling up or down jump lengths of parameter handler
                      
                      acceptance_window = 10,
                      acceptance_target = 0.4,
                      acceptance_band = 0.2,
                      
                      params_handler_hyperparams = { 
                          'initial_jump_lengths': {'couplings' : 0.10,
                                                   'energies' : 0.9,
                                                   'Ls' : 0.010
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
                      
                      iterations_till_progress_update = 20
                      )

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(new_ts := np.linspace(min(ts), max(ts)/1, 1000),
#           quest.initial.calculate_dynamics(new_ts, ['sigmax'])[0])

#%%
best = quest.run()



#%%
best = quest.best
evaluation_ts = np.linspace(ts[0], ts[-1], max(10*len(ts), int(1000)))
best_datasets = best.calculate_dynamics(evaluation_ts, observable_ops = measurement_observables,
                                        custom_function_on_return = False)


#%% chain run outputs:

# output controls bundle:
class Toggles:    
    comparison = True # plot comparison of dynamics
    loss = False # plot cost function progression
    acceptance = False # plot acceptance ratios over subsequenct windows
    graphs = False # plot model graphs with corresponding labels
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    hyperparams = True # save chain hyperparameters as json


# create outputs:
output.Output(toggles = Toggles, filename = filename + '_all_',
       dynamics_ts = [ts, training_ts, evaluation_ts],
       dynamics_datasets = [measurement_datasets, training_measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['all measurements', 'training subset', 'prediction'],
       dynamics_formatting = ['g+', 'b.', 'r-'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       best_loss = quest.best_loss,
       acceptance = quest.chain_windows_acceptance_log,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


# create outputs:
output.Output(toggles = Toggles, filename = filename + '_training_data_gap_',
       dynamics_ts = [np.append(training_ts, ts[-1])],
       dynamics_datasets = [[np.append(training_measurement_datasets[0], 0)]],
       dynamics_datasets_labels = ['measurements'],
       dynamics_formatting = ['g+'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       best_loss = quest.best_loss,
       acceptance = quest.chain_windows_acceptance_log,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )

# create outputs:
output.Output(toggles = Toggles, filename = filename + '_training_data_model_',
       dynamics_ts = [training_ts, evaluation_ts],
       dynamics_datasets = [training_measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['measurements', 'model'],
       dynamics_formatting = ['g+', 'r-'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       best_loss = quest.best_loss,
       acceptance = quest.chain_windows_acceptance_log,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams()
       )


#%% 




