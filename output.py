#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)

"""

import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from definitions import Constants





class Output():
    
    # to do: set font here for all plots
    # each chain can call this and pass it its own things to save...
    # maybe this should be a chain method then? maybe not


    def __init__(self, *, toggles, filename,
                 dynamics_ts = False, dynamics_datasets = [], dynamics_labels = [],
                 cost = [],
                 acceptance_ratios = [],
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
                plt.plot(dynamics_ts*Constants.t_to_sec*1e15, dynamics_datasets[i], colours[i], label = get_label(i))
                plt.xlabel('time (fs)')
                #plt.ylabel('qubit excited population')
                #plt.ylim([0,1.1])
                #HERE
                plt.legend()
                plt.savefig(filename + '_comparison.png', dpi = 1000)
    
    
    
        # plot cost function progression:
        
        if toggles.cost:     
        
            plt.figure()
            plt.plot(cost, 'm-', linewidth = 0.3, markersize = 0.1)
            plt.yscale('log')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            #plt.xlim([0, 10000])
            plt.savefig(filename + '_cost.png', dpi = 1000)
    
    
        
        # save specified model instances (as text and/or pickle and/or graph):
            
        # ensure name selector doesn't go out of bounds:
        def get_model_name(i):
            if not model_names or len(model_names) < len(models_to_save): return '_' + str(i)
            else: return '_' + model_names[i]
            
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
                self.create_model_graph(model,filename + get_model_name(i) + '_graph')
                    
                    
        # save chain hyperparameters dictionary as JSON:      
                    
        if toggles.hyperparams:
            
            def get_chain_name():
                if not chain_name: return ''
                else: return '_' + chain_name    
        
            with open(filename + get_chain_name() + '_hyperparameters.json', 'w') as filestream:
                json.dump(chain_hyperparams,  filestream)
                
                
        
        # plot acceptance ratio evolution:
            
        if toggles.acceptance_ratios:
            
            plt.figure()
            plt.plot(acceptance_ratios, '-', linewidth = 1, markersize = 0.1, color = 'firebrick')
            plt.yscale('linear')
            plt.xlabel('window number')
            plt.ylabel('acceptance ratio')
            #plt.xlim([0, 10000])
            plt.ylim([-0.05,1.05])
            plt.savefig(filename + '_acceptance_ratios.png', dpi = 1000)
            
            
            
    def create_model_graph(self, m, filename):
        
        
        
        G = nx.Graph()
        
        qubit_index, defect_index = 0, 0
        
        ops_labels = {
                     'sigmax': '$\sigma_x$',
                     'sigmaz': '$\sigma_z$',
                     'sigmay': '$\sigma_y$',
                     'sigmam': '$\sigma_-$',
                     'sigmap': '$\sigma_+$',
                    }
        
        normal_vals = {
                       'energy': 5,
                       'coupling': 0.5,
                       'L_offdiag': 0.01,
                       'L_ondiag': 0.01
                      }
        
        
        
        # plot property containers:
        
        node_colours = []
        node_labels = {}
        edge_colours = []
        edge_widths = []
        edge_labels = {}
        node_sizes = []
        
        
        
        # add node for each TLS:
        
        for TLS in m.TLSs:
            
            label = id(TLS)
            
            if TLS.is_qubit: 
                
                # label = 'qubit' + str(qubit_index)
                
                qubit_index += 1
                
                node_colours.append('dodgerblue')
                        
                G.add_node(label, subset = 'qubits')
                
            else:
                
                # label = 'defect' + str(defect_index)
                
                defect_index += 1
                
                node_colours.append('violet')
                
                G.add_node(label, subset = 'defects')
                
            node_labels[label] = np.round(TLS.energy, 2)
            
            node_sizes.append(1000)
                
            
        # add edges for each TLS's couplings and Ls:    
            
        for TLS in m.TLSs:
            
            
            # add couplings - assumed symmetrical here with single label (probably to expand in future):
            
            for partner, couplings in TLS.couplings.items():
                
                for coupling in couplings: # coupling is the tuple
                
                    G.add_edge(id(TLS), id(partner),
                               width = coupling[0]/normal_vals['coupling']*3,
                               label = ops_labels[coupling[1]],
                               colour = 'purple'
                               )
                    
                    edge_labels[(id(TLS), id(partner))] = str(ops_labels[coupling[1]])
                    
                    
            # add Ls:
            
            if True:    
                
                for L, rate in TLS.Ls.items():
                    
                    label = str(id(TLS)) + L
                    
                    G.add_node(label, subset = 'Ls')
                    
                    node_labels[label] = ''#ops_labels[L]
                    
                    node_colours.append('forestgreen')
                    
                    G.add_edge(id(TLS), label,
                               width = rate/normal_vals['L_offdiag'],
                               colour = 'forestgreen'
                               )
                    
                    edge_labels[(id(TLS), label)] = ops_labels[L]
                    
                    node_sizes.append(0)
                    
        
        
        edges = G.edges()           
        edge_widths = [G[u][v]['width'] for u, v in edges]
        edge_colours = [G[u][v]['colour'] for u, v in edges]
           
            
        # different layouts:
        
        # centre_node = 'qubit'  # Or any other node to be in the center
        # defects_nodes = set(G) - {'qubit'}
        # pos = nx.circular_layout(G.subgraph(defects_nodes))
        # pos = nx.circular_layout(G)
        # pos = nx.spring_layout(G, seed = 0)
        # pos = nx.multipartite_layout(G, subset_key = 'subset')
        
        #pos = graphviz_layout(G, prog="twopi")
        #pos = graphviz_layout(G, prog="dot")
        pos = graphviz_layout(G, prog="circo")
        #pos = graphviz_layout(G, prog="sfdp")
        
        
        
        plt.figure()
        plt.title(filename)
        
        nx.draw(G, pos,
                width=edge_widths, edge_color = edge_colours,
                node_color = node_colours, node_size = node_sizes,
                labels = node_labels, font_size = 12
                )
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 12)
        
        
        plt.savefig(filename + '.svg')#, dpi=300)#, bbox_inches='tight')
            
    
    

            
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
    plt.plot(Constants.t_to_sec*ts*1e15, pop_qutip, '-y', label = 'qutip')
    plt.plot(Constants.t_to_sec*ts*1e15, pop_liouvillian, ':k', label = 'liouvillian')
    plt.xlabel('time (fs)')
    plt.ylabel('qubit excited population')
    plt.legend()
    plt.savefig('qutip vs liouvillian comparison.png')
    
    
    # liouvillian colour plot
    
    L = model.LLN
    
    
    # # just to test diagonalisability
    # vals, vecs = np.linalg.eig(L)
    # X = vecs@(np.diag(vals)@(vecs.conjugate().transpose()))
    # plt.figure()
    # plt.matshow(abs(X - np.diag(np.diag(X))), cmap='inferno')
    # plt.title('$|\mathcal{X}|$')
    # plt.colorbar()
    # plt.savefig('X.png', dpi = 1000)
    # plt.show()
    
    
    
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
    



