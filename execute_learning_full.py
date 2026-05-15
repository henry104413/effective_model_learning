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
# order: target_file, experiment_name, defects_count, repetition_number, max_iterations,
# proportion_to_use (of data for training), configuration_number,
# full_switch (3 observables as opposed to 1),
# noise_stdev, shock_anneal_at iterations  
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
    max_iterations = 100

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
    
# noise standard deviation - currently if passed, also reset Metropolis-Hastings temperature to 2*this**2
try:
    noise_stdev = float(sys.argv[9])
except:
    noise_stdev = 0.0

# set iterations at which shock annealing is performed
# ie. when initial_jump_lengths are updated to annealed_jump_lengths
# note: as currently named arguments not supported,
# if subsequent arguments needed, this can be passed as == max_iterations
try:
    shock_anneal_at = int(sys.argv[10])
except:
    shock_anneal_at = False # check!  
    
# set iterations at which tweak width is fixed after adaptation
# note: as currently named arguments not supported,
# if subsequent arguments needed, this can be passed as == max_iterations
try:
    fix_tweak_width_at = int(sys.argv[11])
except:
    fix_tweak_width_at = False # check!
    
# set iterations at which tweak width adaptation begins
# note: as currently named arguments not supported,
# if subsequent arguments needed, this can be passed as == max_iterations
try:
    start_tweak_width_adaptation_at = int(sys.argv[12])
except:
    start_tweak_width_adaptation_at = False # check!
    
# set iterations at which temperature is fixed after adaptation
# note: as currently named arguments not supported,
# if subsequent arguments needed, this can be passed as == max_iterations
try:
    fix_temperature_at = int(sys.argv[13])
except:
    fix_temperature_at = False # check!
    
# set iterations at which temperature adaptation begins
# note: as currently named arguments not supported,
# if subsequent arguments needed, this can be passed as == max_iterations
try:
    start_temperature_adaptation_at = int(sys.argv[14])
except:
    start_temperature_adaptation_at = False # check!

# noise_stdev, shock_anneal_at,
# fix_tweak_width_at, start_tweak_width_adaptation_at,
# fix_temperature_adaptation_at, start_temperature_adaptation_at
# note: latter three - if bash launcher sets that at zero, config file values are taken instead

# get subexperiment name and  corresponding chain configuration:    
subexperiment_name = list(configs.specific_experiment_chain_hyperparams.keys())[configuration_number]
config = copy.deepcopy(configs.default_chain_hyperparams)
for supersede in configs.specific_experiment_chain_hyperparams[subexperiment_name]:
    config[supersede] = configs.specific_experiment_chain_hyperparams[subexperiment_name][supersede]
# note: supersede is name of each hyperparam that is superseded in defaults
# by value for this specific experiment (subexperiment)

# if passed and set above, then populate in config both shock_anneal_at, 
# and temperature (twice the noise variance, ie. 2* noise_stdev**2)
# but NOT if in configs set as tuple in which case temperature will be sampled from inverse gamma distribution
# TO DO: implement passing it here but tricky with  variably 1 or 2 parameter bash argument
if noise_stdev:
    if type(config['temperature_proposal']) in [int, float, bool]:
        config['temperature_proposal'] = 2 * noise_stdev**2
if shock_anneal_at:
    config['shock_anneal_at'] = shock_anneal_at 
if fix_tweak_width_at:
    config['fix_tweak_width_at'] = fix_tweak_width_at 
if start_tweak_width_adaptation_at:
    config['start_tweak_width_adaptation_at'] = start_tweak_width_adaptation_at 
if fix_temperature_at:
    config['fix_temperature_at'] = fix_temperature_at 
if start_temperature_adaptation_at:
    config['start_temperature_adaptation_at'] = start_temperature_adaptation_at 

    
# also FOR NOW append noise stdev to experiment name:
experiment_name = experiment_name + '_std' + str(noise_stdev).replace('.','p')

# run's output files common name:
# example: '250421_Wit4b-grey_ForClusters'
filename = (experiment_name + '_' + target_file.replace('.csv', '') + '_' + subexperiment_name + '_D' + str(defects_count) +
    '_R' + str(repetition_number))
 
print(filename, flush = True)



#%% prepare simulated multi-observable training datasets:
# !!! NOTE: this currently means not using specified target file but taking data from file below instead:

simulated_switch = False

if simulated_switch:    
    
    # note - current use:
    # known noise stdev used to load data,
    # anod to specify Metropolis-Hastings temperature as = 2 * noise VARIANCE;
    # target data filename contains after experiment name this noise stdev as e.g. std0p01, meaning stdev = 0.01
    # - hence now temperature proposal taken as static (as opposed to sampling from a gamma distribution)
        
    # import dictionary of ts, sx, sy, sz observable values 
    # (sx equal to original and rest simulated, all with noise with std = 0.01)   
    noise_level_in_filename = str(noise_stdev).replace('.', 'p') if type(noise_stdev) in [int, float] else ''
    with open('simulated-std' 
              + noise_level_in_filename
              + '_250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best.pickle.csv',
              'rb') as filestream:
        simulated_data = pickle.load(filestream)    
    ts, sx, sy, sz = [simulated_data[x] for x in ['ts', 'sx', 'sy', 'sz']]
            
    
    # measurement data:
    # (encapsulate into lists of datasets and corresponding observable lables)
    if full_switch:
        measurement_datasets = [sx, sy]
        measurement_observables = ['sigmax', 'sigmay']
        print('using full observable set')
    else:
        measurement_datasets = [sx]
        measurement_observables = ['sigmax']
        print('using single observable')
                
else:
    # if this, then using imported experiment data (corresponding to sigma x)
    imported_data = np.genfromtxt(target_file, delimiter=',').transpose()
    ts, sx = imported_data[0], imported_data[1]
    measurement_datasets = [sx]
    measurement_observables = ['sigmax']
    
        
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
                      
                      store_all_proposals = False,
                      
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
    loss = True # loss progression
    log_posterior = True # log posterior 
    log_likelihood_prior = True # superimposed log likelihood and prior
    acceptance_probability = False
    acceptance_windows = True # plot acceptance ratios over subsequenct windows
    graphs = False # plot model graphs with corresponding labels
    pickle = True # save selected models as pickles
    text = True # save selected models as text
    all_proposals = True # save all proposals dictionary as created by chain 
    hyperparams = True # save chain hyperparameters as json


# create outputs - measurements, training subset, prediction on evaluation_ts:
if True:
    output.Output(toggles = Toggles, filename = filename,
       dynamics_ts = [ts, training_ts, evaluation_ts],
       dynamics_datasets = [measurement_datasets, training_measurement_datasets, best_datasets],
       dynamics_datasets_labels = ['all measurements', 'training subset', 'prediction'],
       dynamics_formatting = ['b+', 'b.', 'r-'],
       observable_labels = measurement_observables,
       best_loss = quest.best_loss,
       # !!! TO DO: two lines below are new -- add elsewhere too!
       overall_acceptance = {'parameters tweak': quest.acc_tweak_steps/max(quest.tot_tweak_steps, 1), # avoiding div by 0
                             'reversible jump': quest.acc_RJ_steps/max(quest.tot_RJ_steps, 1)}, # avoiding div by 0
       tweak_widths_after_annealing = quest.tweak_widths_after_annealing,
       windows_acc_rates = quest.windows_acc_rates,
       models_to_save = [best],
       model_names = ['best'],
       chain_hyperparams = quest.get_init_hyperparams(),
       all_proposals = quest.all_proposals,
       shock_anneal_at = config['shock_anneal_at']
       )
    
    




