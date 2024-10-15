#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)

"""

import pickle

import json

import matplotlib.pyplot as plt

from definitions import Constants

# import numpy as np



class Output():
    
    # to do: set font here for all plots
    # each chain can call this and pass it its own things to save...
    # maybe this should be a chain method then? maybe not


    def __init__(self, *, toggles, filename,
                 dynamics_ts = False, dynamics_datasets = [], dynamics_labels = [],
                 cost = [],
                 models_to_save = [],
                 model_names = [],
                 chain_hyperparams = False,
                 chain_name = False,
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
                plt.xlabel('time (s)')
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
    
    
        
        # save specified model instances (as text and/or pickle):
            
        # ensure name selector doesn't go out of bounds:
        def get_model_name(i):
            if not model_names or len(model_names) < len(models_to_save): return '_' + str(i)
            else: return '_' + model_names[i]
            
        for i, model in enumerate(models_to_save):
            
            # as pickle:
        
            if toggles.pickle:
                with open(filename + get_model_name(i) +'.pickle', 'wb') as filestream:
                    pickle.dump(models_to_save[i],  filestream)
                
            # as text
            
            if toggles.text:
                with open(filename + get_model_name(i) + '.txt', 'w') as filestream:
                    filestream.write(models_to_save[i].model_description_str())
                    
                    
                    
        # save chain hyperparameters dictionary as JSON:      
                    
        if toggles.hyperparams:
            
            def get_chain_name():
                if not chain_name: return ''
                else: return '_' + chain_name    
        
            with open(filename + get_chain_name() + '_hyperparameters.json', 'w') as filestream:
                json.dump(chain_hyperparams,  filestream)
            
            
            
            
def compare_qutip_Liouvillian(model, ts):
    
    
    import numpy as np        
    import matplotlib.pyplot as plt
    import matplotlib.colors as colour
    from matplotlib import colormaps
    from definitions import Constants
    
    
    pop_qutip = model.calculate_dynamics(ts, dynamics_method = 'qutip')
    
    pop_liouvillian = model.calculate_dynamics(ts, dynamics_method = 'liouvillian')
    
    
    #%% 
    # ad hoc plots:
        
    
    
    # qutip vs liouvillian dynamics
    plt.figure()
    plt.plot(Constants.t_to_sec*ts, pop_qutip, '-y', label = 'qutip')
    plt.plot(Constants.t_to_sec*ts, pop_liouvillian, ':k', label = 'liouvillian')
    plt.xlabel('time (fs)')
    plt.ylabel('qubit excited population')
    plt.legend()
    plt.savefig('qutip vs liouvillian comparison.png')
    
    
    # liouvillian colour plot
    
    L = model.LLN
    
    
    plt.figure()
    plt.matshow(abs(L-np.diag(np.diag(L))), cmap='inferno')
    plt.title('$|\mathcal{L}|$ off diagonal')
    plt.colorbar()
    plt.savefig('off diag.png', dpi = 1000)
    plt.show()
    
    
    plt.figure()
    cmap = colormaps['inferno'].copy()
    cmap.set_bad('k', alpha=1)
    plt.matshow(abs(L), cmap=cmap, norm=colour.LogNorm(0.005, 100), interpolation = 'none')
    plt.title('$|\mathcal{L}|$')
    plt.colorbar()
    plt.savefig('liouvillian.png', dpi = 1000)
    plt.show()
    



       