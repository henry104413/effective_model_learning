#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)

"""

from __future__ import annotations
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import typing
import copy
import pprint

import definitions

if typing.TYPE_CHECKING:
    from basic_model import BasicModel
    from learning_model import LearningModel

# shorthands:
K_to_eV = definitions.Constants.K_to_eV
t_to_sec = definitions.Constants.t_to_sec
# note: h_bar=1, e=1

# fontsize in figures (default was 10)
plt.rcParams["font.size"] = 20


class Output:
    
    """
    Instantiation immeadiately creates outputs as specified by initialiser arguments.
    """

    def __init__(self, *, 
                 toggles: type,
                 filename: str,
                 dynamics_ts: np.ndarray = False, 
                 dynamics_datasets: list[list[np.ndarray]] = None, 
                 dynamics_datasets_labels: list[str] = None,
                 dynamics_formatting: list[str] = False,
                 observable_labels: list[str] = None,
                 best_loss: float = None,
                 windows_acc_rates: list[float|int] = None,
                 overall_acceptance: dict[str,float] = False,
                 tweak_widths_after_annealing: dict[str,float] = False,
                 models_to_save: list[BasicModel|LearningModel] = None,
                 model_names: list[str] = None,
                 all_proposals: dict[str, list[float]|list[LearningModel]] = None,
                 shock_anneal_at: int = False,
                 chain_hyperparams: dict = False,
                 chain_name: str = False,
                 fontsize: float = False):
        """
        Creates outputs as specified by arguments.
        """
        
        if dynamics_datasets is None: dynamics_datasets = [], 
        if dynamics_datasets_labels is None: dynamics_datasets_labels = [],
        if observable_labels is None: observable_labels = [],
        if windows_acc_rates is None: windows_acc_rates = [],
        if models_to_save is None: models_to_save = [],
        if model_names is None: model_names = [],
        if all_proposals is None: all_proposals = {},
        
        self.fontsize = fontsize
        
        plt.rcParams['agg.path.chunksize'] = 101

        
        # save as json overall acceptance for different step types
        if overall_acceptance:
            with open(filename + '_overall_acceptance.json', 'w') as filestream:
                json.dump(overall_acceptance, filestream)
            # HERE EXPAND WITH DETAILED ACCEPTANCE FOR DIFFERENT STEP CLASSES AND ANNEALING STAGES
            # USING BELOW AND SIMILAR
            # [True for (x,y,z) in 
            #  zip(all_proposals['acceptance'], all_proposals['step_types'], all_proposals['annealed'])
            #  if x and (y not in ['tweak all parameters', 'jump to best']) and z]
        
        # save json of tweak widths after annealing (dependent on annealing choice and any adaptation)
        tweak_widths_after_annealing
        if tweak_widths_after_annealing:
            with open(filename + '_tweak_widths_after_annealing.json', 'w') as filestream:
                json.dump(tweak_widths_after_annealing, filestream)
        
    
        # save specified model instances (as text and/or pickle):
            
        # returns model name if available or its number otherwise:
        def get_model_name(i):
            if not model_names or len(model_names) < len(models_to_save): return '_' + str(i)
            else: return '_' + model_names[i]
        
        # save each model as per toggles:    
        for i, model in enumerate(models_to_save):
            # as pickle:
            if toggles.pickle:
                with open(filename + get_model_name(i) +'.pickle', 'wb') as filestream:
                    pickle.dump(models_to_save[i],  filestream)
            # as text:
            if toggles.text:
                with open(filename + get_model_name(i) + '.txt', 'w') as filestream:
                    filestream.write(models_to_save[i].model_description_str())
            # as graphs:
            if toggles.graphs:
                self.create_model_graph(model, filename + get_model_name(i) + '_graph')
       
        
        # save all proposals into single pickle:
        # ie. dictionary with keys: acceptance, log_likelihood_prior, step_types, shock_anneal_at, vectors
        # and if chain not in lean_mode: proposals, loss, acceptance_probability, annealed, log_posterior;
        # - proposals are instances of LearningModel saved if save_all_propoals was on
        if toggles.all_proposals:
            with open(filename + '_proposals.pickle', 'wb') as filestream:
                pickle.dump(all_proposals, filestream)

        
        # save chain hyperparameters dictionary as dictionary string and as pickle:
        # note: unfortunately JSON doesn't support tuples as keys
        # note: apparently string can be loaded back using ast.literal_eval
        if toggles.hyperparams:
            def get_chain_name():
                if not chain_name: return ''
                else: return '_' + chain_name    
            with open(filename + get_chain_name() + '_hyperparameters.txt', 'w') as filestream:
                filestream.write(pprint.pformat(chain_hyperparams))
            with open(filename + get_chain_name() + '_hyperparameters.pickle', 'wb') as filestream:
                pickle.dump(chain_hyperparams,  filestream)
                
        
        # plot comparison of selected (usually best) dynamics datasets (up to 4) wrt all observables:
        if toggles.comparison:
            
            # format strings for different data sets if specified, else default for up to 4: 
            if isinstance(y := dynamics_formatting, list) and bool(y) and all(isinstance(x, str) for x in y):
                line_formats = dynamics_formatting
            else:
                line_formats = ['bo', 'r--', 'k-.', 'g:'] 
            
            #ts = 1e15*t_to_sec*dynamics_ts # dynamics times in fs
            
            # returns dynamics times for each datasets:
            # either same array if one passed, or corresponding element if list of arrays passed
            # array dimensions matching dataset dimensions not checked
            def get_dynamics_times(i):
                if isinstance(dynamics_ts, np.ndarray): 
                    return dynamics_ts
                elif (isinstance(dynamics_ts, list) and bool(dynamics_ts) 
                      and all(isinstance(x, np.ndarray) for x in dynamics_ts)):
                    return dynamics_ts[i]
                else:
                    raise RuntimeError('Times for plotting dynamics datasets not specified correctly')
            
            # returns corresponding dynamics dataset label including checking label available:
            def get_dynamics_dataset_label(i):
                if not dynamics_datasets_labels or len(dynamics_datasets_labels) < len(dynamics_datasets): return None
                else: return dynamics_datasets_labels[i]
            
            # plot comparison for each observables:
            for i, observable in enumerate(observable_labels):
                plt.figure()
                plt.ylabel(definitions.observable_shorthand2pretty[observable])
                #plt.ylabel(r'<$\sigma_x$>')
                plt.xlabel(r'time ($\mu s$)')
                plt.ylim([-1.1,1.1])
                #r'Microstrain [$\mu \epsilon$]'
                    
                # plot all the datasets in the comparison for this observable:
                # assumed times may differ for datasets but same across each dataset for all observables
                for j, dataset in enumerate(dynamics_datasets):    
                    plt.plot(get_dynamics_times(j), dataset[i], line_formats[j], label = get_dynamics_dataset_label(j)
                             ,markersize = 5, markeredgewidth = 1, linewidth = 2
                             )
                            
                plt.legend(fontsize = 14)
                if not False:
                    plt.text(0, plt.gca().get_ylim()[0] + (plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0])/50,
                         'best loss = ' + '{:.2e}'.format(best_loss))
                try:
                    plt.savefig(filename + '_' + observable + '_comparison.svg', dpi = 1000, bbox_inches='tight')
                except Exception as exception:
                    print('Error when saving dynamics comparison plot:\n' + str(exception))
                plt.clf()
                    
        

        # chain trackers:
        
            
        # loss for all proposals:
        if toggles.loss and 'loss' in all_proposals:
            plt.figure()
            plt.plot(all_proposals['loss'], 'm-', linewidth = 0.3, markersize = 0.1)
            plt.yscale('log')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.text(0, #(plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0])/20,
                     10**(0.98*np.log10(plt.gca().get_ylim()[0])),
                     'best loss = ' + '{:.2e}'.format(best_loss))
            #plt.xlim([0, 10000])
            try:
                plt.savefig(filename + '_loss.svg', dpi = 1000, bbox_inches='tight')
            except Exception as exception:
                print('Error when saving loss progression plot:\n' + str(exception))
            with open(filename + '_loss.pickle', 'wb') as filestream:
                pickle.dump(all_proposals['loss'], filestream)
                
                
        # iterables with entries for each annealing stage - if both annealed and unannealed present,
        # also plot and save these separatenly (only for accepted stuff);
        # if all proposals dictionary contains annealed boolean list, iterate over that,
        # otherwise use generator that flips upon reaching shock_anneal_at if passed
        annealing_stages_labels = ['unannealed', 'annealed', ''] # last means whole chain regardless of annealing
        if 'annealed' in all_proposals:
            annealing_stages_switch = [temp := (any(all_proposals['annealed'])
                                                and not all(all_proposals['annealed'])),
                                       temp, True]
            def annealed_generator(): return (x for x in all_proposals['annealed'])
        elif (type(shock_anneal_at) == int): 
            annealing_stages_switch = [temp := (shock_anneal_at > 0
                                                and shock_anneal_at < len(all_proposals['acceptance'])-1),
                                       temp, True]
            def annealed_generator(): return (x >= shock_anneal_at 
                                              for x in range(len(all_proposals['acceptance'])))
        else:
            annealing_stages_switch = [False, False, True]
            def annealed_generator(): return (True for x in all_proposals['acceptance'])
        def annealing_condition(stage_label: str,
                                proposal_annealing_flag: bool):
            if stage_label == 'unannealed': return (not proposal_annealing_flag)
            elif stage_label == 'annealed': return proposal_annealing_flag
            elif stage_label == '': return True
            
        for q, stage_label in enumerate(annealing_stages_labels):  

            if not annealing_stages_switch[q]: continue  
            if stage_label: 
                stage_label_filenames = '_' + stage_label
                stage_label_axislabels = ' ' + stage_label
            else: 
                stage_label_filenames = stage_label                                                       
                stage_label_axislabels = stage_label
        
            # plot loss progression over chain steps, also save list as pickle:
            if toggles.loss and 'loss' in all_proposals:     
                # also loss of just accepted models:    
                accepted_loss = [x for (x, y, z) in 
                                 zip(all_proposals['loss'][1:], all_proposals['acceptance'], annealed_generator())
                                 if y and annealing_condition(stage_label, z)]
                if accepted_loss: best_loss = min(accepted_loss) # check for empty sequence
                else: best_loss = 0
                plt.figure()
                plt.plot(accepted_loss, '-', c = 'orange', linewidth = 0.3, markersize = 0.1)
                plt.yscale('log')
                plt.xlabel('accepted' + stage_label_axislabels + ' proposal no.')
                plt.ylabel('loss')
                if not stage_label:
                    plt.text(0, #(plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0])/20,
                             10**(0.98*np.log10(plt.gca().get_ylim()[0])),
                             'best loss = ' + '{:.2e}'.format(best_loss))
                try: 
                    plt.savefig(filename + stage_label_filenames + '_accepted_loss.svg', dpi = 1000, bbox_inches='tight')
                except Exception as exception:
                    print('Error when saving accepted loss progression plot:\n' + str(exception))
                with open(filename + stage_label_filenames + '_accepted_loss.pickle', 'wb') as filestream:
                    pickle.dump(accepted_loss, filestream)
                plt.clf()
                del accepted_loss
                          
            # plot accepted log-posterior progression over chain steps, also save list as pickle:
            if toggles.log_posterior and 'log_posterior' in all_proposals:     
                    
                accepted_log_posterior = [x for (x, y, z) in 
                                          zip(all_proposals['log_posterior'][1:], all_proposals['acceptance'], annealed_generator())
                                          if y and annealing_condition(stage_label, z)]
                if accepted_log_posterior: best_log_posterior = min(accepted_log_posterior) # check for empty sequence
                else: best_log_posterior = 0
                plt.figure()
                plt.plot(accepted_log_posterior, '-', c = 'turquoise', linewidth = 0.3, markersize = 0.1)
                plt.yscale('symlog')
                plt.xlabel('accepted' + stage_label_axislabels + ' proposal no.')
                plt.ylabel('log posterior')
                # plt.text(0, #(plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0])/20,
                #          10**(0.98*np.log10(plt.gca().get_ylim()[0])),
                #          'best log posterior = ' + '{:.2e}'.format(best_log_posterior))
                try: 
                    plt.savefig(filename + stage_label_filenames + '_accepted_log_posterior.svg', dpi = 1000, bbox_inches='tight')
                except Exception as exception:
                    print('Error when saving accepted log posterior progression plot:\n' + str(exception))
                with open(filename + stage_label_filenames + '_accepted_log_posterior.pickle', 'wb') as filestream:
                    pickle.dump(accepted_log_posterior, filestream)
                plt.clf()
                del accepted_log_posterior
            
            # plot log-likelihood-prior progression over chain steps, also save list as pickle:
            if toggles.log_likelihood_prior and 'log_likelihood_prior' in all_proposals:     
            
                accepted_log_likelihood_prior = [x for (x, y, z) in 
                                          zip(all_proposals['log_likelihood_prior'][1:], all_proposals['acceptance'], annealed_generator())
                                          if y and annealing_condition(stage_label, z)]
                accepted_log_likelihood = [x for (x,y) in accepted_log_likelihood_prior]
                accepted_log_prior = [y for (x,y) in accepted_log_likelihood_prior]
                plt.figure()
                plt.plot(accepted_log_likelihood, '-', c = 'red', linewidth = 0.3, markersize = 0.1, label = 'likelihood', alpha = 0.6)
                plt.plot(accepted_log_prior, '-', c = 'green', linewidth = 0.3, markersize = 0.1, label = 'prior', alpha = 0.6)
                plt.yscale('symlog')
                plt.xlabel('accepted' + stage_label_axislabels + ' proposal no.')
                plt.ylabel('log posterior')
                plt.legend()
                try: 
                    plt.savefig(filename + stage_label_filenames + '_accepted_log_likelihood_prior.svg', dpi = 1000, bbox_inches='tight')
                except Exception as exception:
                    print('Error when saving accepted log likelihood and prior progression plot:\n' + str(exception))
                with open(filename + stage_label_filenames + '_accepted_log_likelihood_prior.pickle', 'wb') as filestream:
                    pickle.dump(accepted_log_likelihood_prior, filestream)
                plt.clf() 
                del accepted_log_likelihood_prior
       
        
        # plot acceptance probability progression:
        if toggles.acceptance_probability and 'acceptance_probability' in all_proposals:
            plt.figure()
            plt.plot(all_proposals['acceptance_probability'], 'b-', linewidth = 0.3, markersize = 0.1)
            plt.xlabel('iteration')
            plt.ylabel('acceptance probability')
            plt.ylim([0,2])
            try:
                plt.savefig(filename + '_AP.svg', dpi = 1000, bbox_inches='tight')
            except Exception as exception:
                print('Error when saving acceptance probability progression plot:\n' + str(exception))
            with open(filename + '_AP.pickle', 'wb') as filestream:
                pickle.dump(all_proposals['acceptance_probability'], filestream)
            plt.clf()
            
            # also acceptance_probability of just accepted models:    
            accepted_AP = [x for (x, y) in zip(all_proposals['acceptance_probability'][:], all_proposals['acceptance']) if y]
            plt.figure()
            plt.plot(accepted_AP, '-', c = 'orange', linewidth = 0.3, markersize = 0.1)
            plt.xlabel('accepted proposal no.')
            plt.ylabel('acceptance probability')
            plt.ylim([0,2])
            try:
                plt.savefig(filename + '_accepted_AP.svg', dpi = 1000, bbox_inches='tight')
            except Exception as exception:
                print('Error when saving accepted acceptance probability progression plot:\n' + str(exception))
            with open(filename + '_accepted_AP.pickle', 'wb') as filestream:
                      pickle.dump(accepted_AP, filestream)
            plt.clf()
            
            
        # plot acceptance ratio evolution:
        if toggles.acceptance_windows and windows_acc_rates:
            
            # overlay of acceptance rates for different type steps:
            plt.figure()
            plt.plot(windows_acc_rates['RJ'], '.-', linewidth = 0.1, markersize = 0.5, color = 'firebrick', label = 'RJ')
            plt.plot(windows_acc_rates['tweak'], '.-', linewidth = 0.1, markersize = 0.5, color = 'limegreen', label = 'tweak')
            plt.plot(windows_acc_rates['total'], '.-', linewidth = 0.1, markersize = 0.5, color = 'mediumblue', label = 'total')
            plt.yscale('linear')
            plt.xlabel('window number')
            plt.ylabel('acceptance ratio')
            #plt.xlim([0, 10000])
            plt.ylim([-0.05,1.05])
            plt.legend()
            try:
                plt.savefig(filename + '_acceptance.svg', dpi = 1000, bbox_inches='tight')
            except Exception as exception:
                print('Error when saving acceptance over windows plot:\n' + str(exception))
            plt.clf()
            
            # corresponding temperatures:
            plt.figure()
            plt.plot(windows_acc_rates['temperature'], '.-', linewidth = 0.1, markersize = 0.5, color = 'navy', label = 'temperature')
            plt.yscale('log')
            plt.xlabel('window number')
            plt.ylabel('temperature')
            try:
                plt.savefig(filename + '_temperature.svg', dpi = 1000, bbox_inches='tight')
            except Exception as exception:
                print('Error when saving temperature over windows plot:\n' + str(exception))
            plt.clf()
            
            # corresponding tweak widths (log scaling for now):
            plt.figure()
            colours = ['orange', 'yellowgreen', 'orchid']
            linestyles = ['.-', '.-', '+--']
            for i, key in enumerate(['Ls', 'couplings', 'energies']):
                plt.plot(windows_acc_rates['tweak widths'][key]
                         , linestyles[i], linewidth = 0.1, markersize = 0.5, color = colours[i], label = key)
            plt.yscale('log')
            plt.legend()
            plt.xlabel('window number')
            plt.ylabel('tweak width')
            try:
                plt.savefig(filename + '_widths.svg', dpi = 1000, bbox_inches='tight')
            except Exception as exception:
                print('Error when saving tweak widths over windows plot:\n' + str(exception))
            plt.clf()
            
                        
            
    