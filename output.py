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
import networkx as nx
import typing

import definitions

if typing.TYPE_CHECKING:
    from basic_model import BasicModel
    from learning_model import LearningModel

# shorthands:
K_to_eV = definitions.Constants.K_to_eV
t_to_sec = definitions.Constants.t_to_sec
# note: h_bar=1, e=1



class Output:
    
    """
    Instantiation immeadiately creates outputs as specified by initialiser arguments.
    """

    def __init__(self, *, 
                 toggles: type,
                 filename: str,
                 dynamics_ts: np.ndarray = False, 
                 dynamics_datasets: list[list[np.ndarray]] = [], 
                 dynamics_datasets_labels: list[str] = [],
                 observable_labels: list[str] = [],
                 loss: list[float|int] = [],
                 acceptance: list[float|int] = [],
                 models_to_save: list[BasicModel|LearningModel] = [],
                 model_names: list[str] = [],
                 chain_hyperparams: dict = False,
                 chain_name: str = False,
                 fontsize: float = False):
        """
        Creates outputs as specified by arguments.
        """
        
        self.fontsize = fontsize
        
        
        # plot comparison of dynamics datasets (up to 4) wrt all observables:
        if toggles.comparison:
        
            line_formats = ['b-', 'r--', 'k-.', 'g:'] # different data set plot formats
            # note: now supports max 4 data sets
            ts = 1e15*t_to_sec*dynamics_ts # dynamics times in seconds
            
            # returns corresponding danamics dataset label including checking label available:
            def get_dynamics_dataset_label(i):
                if not dynamics_datasets_labels or len(dynamics_datasets_labels) < len(dynamics_datasets): return None
                else: return dynamics_datasets_labels[i]
            
            # plot comparison for each observables:
            for i, observable in enumerate(observable_labels):
                plt.figure()
                plt.ylabel(observable)
                plt.xlabel('time (fs)')
                
                # plot all the datasets in the comparison for this observable:
                for j, dataset in enumerate(dynamics_datasets):    
                    plt.plot(ts, dataset[i], line_formats[j], label = get_dynamics_dataset_label(j))
                            
                plt.legend()
                plt.savefig(filename + '_' + observable + '_comparison.svg', dpi = 1000)
                 
    
        # plot loss progression over chain steps:
        if toggles.loss:     
            plt.figure()
            plt.plot(loss, 'm-', linewidth = 0.3, markersize = 0.1)
            plt.yscale('log')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            #plt.xlim([0, 10000])
            plt.savefig(filename + '_loss.svg', dpi = 1000)
            
            
        # plot acceptance ratio evolution:
        if toggles.acceptance:
            plt.figure()
            plt.plot(acceptance, '-', linewidth = 1, markersize = 0.1, color = 'firebrick')
            plt.yscale('linear')
            plt.xlabel('window number')
            plt.ylabel('acceptance ratio')
            #plt.xlim([0, 10000])
            plt.ylim([-0.05,1.05])
            plt.savefig(filename + '_acceptance.svg', dpi = 1000)
        
    
    
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
        
        graphviz_layout = nx.drawing.nx_pydot.graphviz_layout
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

