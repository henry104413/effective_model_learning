#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)

"""

import pickle

import matplotlib.pyplot as plt

from definitions import Constants

# import numpy as np



class Output():
    
    # to do: set font here for all plots


    def __init__(self, *, toggles, filename,
                 dynamics_ts = False, dynamics_datasets = [], dynamics_labels = [],
                 cost = [],
                 models_to_save = [],
                 model_names = [],
                 fontsize = False):
        
        
        
        self.fontsize = fontsize
        
        
    
        # plot dynamics comparison (up to 4):
            
        if toggles.comparison:
            
            colours = ['r-', 'b--', 'k:', 'g-.']
            
            # ensure label selector doesn't go out of bounds
            def get_label(i):
                if not dynamics_labels or len(dynamics_labels) < len(dynamics_datasets): return None
                else: return dynamics_labels[i]
            
            plt.figure()
            
            for i in range(min(len(dynamics_datasets), 4)):    
                plt.plot(dynamics_ts*Constants.t_to_sec, dynamics_datasets[i], colours[i], label = get_label(i))
                plt.xlabel('time (fs)')
                plt.ylabel('qubit excited population')
                plt.ylim([0,1.1])
                plt.legend()
                plt.savefig(filename + '_comparison.svg')
    
    
    
        # plot cost function progression:
        
        if toggles.cost:     
        
            plt.figure()
            plt.plot(cost, 'm-', linewidth = 0.1, markersize = 0.1)
            plt.yscale('log')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.savefig(filename + '_cost.svg')
    
    
        
        # save model instances (as text and/or pickle as specified):
            
        # ensure name selector doesn't go out of bounds:
        def get_model_name(i):
            if not model_names or len(model_names) < len(models_to_save): return '_' + str(i)
            else: return model_names[i]
            
        for i, model in enumerate(models_to_save):
            
            # as pickle:
        
            if toggles.pickle:
                with open(filename + get_model_name(i) +'.pickle', 'wb') as filestream:
                    pickle.dump(models_to_save[i],  filestream)
                
            # as text
            
            if toggles.text:
                with open(filename + get_model_name(i) + '.txt', 'w') as filestream:
                    filestream.write(models_to_save[i].description())
         