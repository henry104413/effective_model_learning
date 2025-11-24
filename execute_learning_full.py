#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""


import numpy as np
import sys # for passing command line arguments
import time
import copy
import pickle
# note: import os also called below in case of an exception


import basic_model
import learning_chain
import output
import configs
import definitions


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
# note: includes custom experiment name and target datafile name
try:
    experiment_name = str(sys.argv[2])
except:
    experiment_name = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())
    
# set number of defects,
try:
	defects_count = int(sys.argv[3])
except: 
	defects_count = 1
	
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
    max_iterations = 1000

# set proportion (ratio) of available data values to use for training:
# note: currently taken from start and same for all data sets; 1 means use all
# note: must be a number and not expression
try:
    proportion_to_use = float(sys.argv[6])
except:
    proportion_to_use = 1
    
# set chain hyperparameter configuration number for specific subexperiment:
# note: assumes configurations stored as LEarningChain initialiser keyword argument dictionaries
# imported from configs file where specific_experiment_chain_hyperparams at least this number of entries
try:
    configuration_number = int(sys.argv[7])
except:
    configuration_number = False

# set switch for training on full set:
# if 1 (or other true) uses sx, sy, sz from imported simulated data pickle,
# if 0 (false when boolified) uses just sx (equal to original data):
try:
    full_switch = bool(int(sys.argv[8]))
except:
    full_switch = True    

# get subexperiment name and  corresponding chain configuration:    
subexperiment_name = list(configs.specific_experiment_chain_hyperparams.keys())[configuration_number]
config = copy.deepcopy(configs.default_chain_hyperparams)
for supersede in configs.specific_experiment_chain_hyperparams[subexperiment_name]:
    config[supersede] = configs.specific_experiment_chain_hyperparams[subexperiment_name][supersede]
# note: supersede is name of each hyperparam that is superseded in defaults
# by value for this specific experiment (subexperiment)

# run's output files common name:
# example: '250421_Wit4b-grey_ForClusters'
filename = (experiment_name + '_' + target_file.replace('.csv', '') + '_' + subexperiment_name + '_D' + str(defects_count) +
    '_R' + str(repetition_number))
 
print(filename, flush = True)



#%% prepare simulated multi-observable training datasets:
    
# import dictionary of ts, sx, sy, sz observable values 
# (sx equal to original and rest simulated, all with noise with std = 0.01)    
with open('simulated_250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle',
          'rb') as filestream:
    simulated_data = pickle.load(filestream)    
ts, sx, sy, sz = [simulated_data[x] for x in ['ts', 'sx', 'sy', 'sz']]
        

# measurement data:
# (encapsulate into lists of datasets and corresponding observable lables)
if full_switch:
    measurement_datasets = [sx, sy, sz]
    measurement_observables = ['sigmax', 'sigmay', 'sigmaz']
    print('using full observable set')
else:
    measurement_datasets = [sx]
    measurement_observables = ['sigmax']
    print('using single observable')
            
# times and measurement data to use for training:
# (encapsulate into lists of datasets and corresponding observable lables:)
# note: currently here not training on subset but can be implemented like below:
# training_ts = ts[:int(proportion_to_use*len(ts))]
training_ts = ts
training_measurement_datasets = measurement_datasets
training_measurement_observables = measurement_observables



#%% AD HOC: load best to use as initial for testing:

# with open('250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle',
#           'rb') as filestream:
#     initial_model = pickle.load(filestream)

#%% perform learning:

# if qubit initial state required:
qubit_initial_state = definitions.ops['plus']
defect_initial_state = definitions.ops['mm']    



# instance of learning (quest for best model):
quest = learning_chain.LearningChain(target_times = training_ts,
                      target_datasets = training_measurement_datasets,
                      target_observables = training_measurement_observables,
                      
                      initial = (1, defects_count), # (qubit energy, number of defects)
                      #initial = initial_model,
                      qubit_initial_state = qubit_initial_state,
                      defect_initial_state = definitions.ops['mm'],    

                      max_chain_steps = max_iterations,
                      
                      store_all_proposals = True,
                      
                      **config # specific experiment chain hyperparameters
                      
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
    loss = True # plot cost function progression
    log_posterior = True # plot 1) log posterior and 2) superimposed log likelihood and prior
    acceptance_probability = False
    acceptance = True # plot acceptance ratios over subsequenct windows
    graphs = False # plot model graphs with corresponding labels
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    all_proposals = True # save all proposals in dictionary of lists under keys 'losses', 'proposals' 
    hyperparams = True # save chain hyperparameters as json


# create outputs - measurements, training subset, prediction on evaluation_ts:
if True:
    output.Output(toggles = Toggles, filename = filename,
       dynamics_ts = [ts, training_ts, evaluation_ts],
       dynamics_datasets = [measurement_datasets, training_measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['all measurements', 'training subset', 'prediction'],
       dynamics_formatting = ['b+', 'b.', 'r-'],
       observable_labels = measurement_observables,
       loss = quest.explored_loss,
       best_loss = quest.best_loss,
       # !!! TO DO: two lines below are new -- add elsewhere too!
       acceptance_probability = quest.explored_acceptance_probability,
       overall_acceptance = {'parameters tweak': quest.acc_tweak_steps/max(quest.tot_tweak_steps, 1), # avoiding div by 0
                             'reversible jump': quest.acc_RJ_steps/max(quest.tot_RJ_steps, 1)}, # avoiding div by 0
       acceptance = quest.chain_windows_acceptance_log,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams(),
       all_proposals = quest.all_proposals
       )


# create outputs - only training subset of measurements, last datapoint plotted for consistent x-axis scaling:
if False:
    output.Output(toggles = Toggles, filename = filename + '_training_data_gap',
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
       chain_hyperparams = quest.get_init_hyperparams(),
       all_proposals = quest.all_proposals
       )

# create outputs - training on training_ts (possibly subset) and evaluation on best_datasets:
if False:
    output.Output(toggles = Toggles, filename = filename + '_training_data_model',
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
       chain_hyperparams = quest.get_init_hyperparams(),
       all_proposals = quest.all_proposals
       )


#%% 






