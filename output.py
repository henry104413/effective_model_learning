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
                 dynamics_formatting: list[str] = False,
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
                plt.ylabel(observable)
                plt.xlabel('time (fs)')
                
                # plot all the datasets in the comparison for this observable:
                # assumed times may differ for datasets but same across each dataset for all observables
                for j, dataset in enumerate(dynamics_datasets):    
                    plt.plot(get_dynamics_times(j), dataset[i], line_formats[j], label = get_dynamics_dataset_label(j))
                            
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
                
                
            
    def create_model_graph(self,
                           model: BasicModel|LearningModel,
                           filename: str
                           ) -> None:
        
        """
        Creates and saves in current folder a network graph of argument model.
        """
        
        #  auxiliary functions to produce graph labels:
        
        def op_symbol(op: str) -> str:    
            """
            Returns matplotlib friendly string with symbol
            corresponding to argument operator label.
            """
            match op:
                case 'sigmax': return r'$\sigma_x$'
                case 'sigmaz': return r'$\sigma_z$'
                case 'sigmay': return r'$\sigma_y$'
                case 'sigmam': return r'$\sigma_-$'
                case 'sigmap': return r'$\sigma_+$'
                case _: return op

        def make_coupling_edge_label(coupling: tuple[int|float, list[tuple[str]]]) -> str:
            """
            Returns string label for argument single coupling given as
            (strength, [('op_here', 'op_there'), ...]).
            """
            temp = ''
            temp = temp + str(np.round(coupling[0], 2)) + r'$\times$' + '('
            first_iteration = True
            for op_pair in coupling[1]: 
                # go over all operator pairs of this coupling, ie. tuples (op1_label, op2_label)
                if not first_iteration:
                    temp = temp + ' + '
                first_iteration = False    
                temp = temp + op_symbol(op_pair[0]) + r'$\otimes$' + op_symbol(op_pair[1])
            temp = temp + ')'
            return temp


        
        # reference values to determine graph line thickness and marker size:
        normal_vals = {
                       'energy': 5,
                       'coupling': 0.5,
                       'L_offdiag': 0.01,
                       'L_ondiag': 0.01
                      }
        
        # initialise graph and corresponding containers:
        #G = nx.Graph()
        G = nx.MultiGraph()
        node_colours = []
        node_labels = {}
        edge_colours = []
        edge_widths = []
        edge_labels = {}
        node_sizes = []
        
        # add node for each TLS:
        qubit_index, defect_index = 0, 0
        for TLS in model.TLSs:
            label = id(TLS)
            if TLS.is_qubit: # ie. qubit
                # label = 'qubit' + str(qubit_index)
                qubit_index += 1
                node_colours.append('dodgerblue')
                G.add_node(label, subset = 'qubits')
            elif not TLS.is_qubit: # ie. defect
                # label = 'defect' + str(defect_index)
                defect_index += 1
                node_colours.append('violet')
                G.add_node(label, subset = 'defects')
            node_labels[label] = np.round(TLS.energy, 2)
            node_sizes.append(1000)
            
        #print(G.nodes(data=True))
        
        # add edges for each TLS's couplings and Ls:    
        for TLS in model.TLSs:
            
            # add couplings:
            # storage: {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...])]}
            # # eg.
            # {'defect1': [(0.6, [('sigmap', 'sigmam'), ('sigmam', 'sigmap')]), 
            #                          (0.7, [('sigmax', 'sigmax')])]
            # # this must be reworked
            for partner, couplings in TLS.couplings.items(): # couplings is a list of tuples of (str, [(op_here, op_there),...])
                
                for coupling in couplings: # coupling tuple is (str, [(op_here, op_there),...])
                
                    G.add_edge(id(TLS), id(partner),
                               width = coupling[0]/normal_vals['coupling']*3,
                               label = make_coupling_edge_label(coupling),
                               colour = 'purple'
                               )
                    
                    # edge_labels[(id(TLS), id(partner))] = make_coupling_edge_label(coupling)
                    # this line might be a problem - multiple edges should now exist between the two nodes
                    
            #G=nx.MultiGraph ()
            # G.add_edge(1,2,weight=1)
            # G.add_edge(1,2,weight=2)
            # G.add_edge(1,2,weight=3)
            # G.add_edge(3,1,weight=4)
            # G.add_edge(3,2,weight=5)
            # for edge in G.edges(data=True): edge[2]['label'] = edge[2]['weight']
            # node_label = nx.get_node_attributes(G,'id')
            # pos = nx.spring_layout(G)
            # node_label = nx.get_node_attributes(G,'id')
            # pos = nx.spring_layout(G)
            # from IPython.display import Image
            # p=nx.drawing.nx_pydot.to_pydot(G)
            # p.write_png('multi.png')
            # Image(filename='multi.png')
            
            # add Ls:
            for L, rate in TLS.Ls.items():
                break
                label = str(id(TLS)) + L
                G.add_node(label, subset = 'Ls')
                node_labels[label] = ''#ops_labels[L]
                node_colours.append('forestgreen')
                G.add_edge(id(TLS), label,
                           width = rate/normal_vals['L_offdiag'],
                           colour = 'forestgreen'
                           )
                edge_labels[(id(TLS), label)] = op_symbol(L)
                node_sizes.append(0)
                
                
                
        def draw_labeled_multigraph(G, attr_name, ax=None):
            """
            Length of connectionstyle must be at least that of a maximum number of edges
            between pair of nodes. This number is maximum one-sided connections
            for directed graph and maximum total connections for undirected graph.
            """
            # Works with arc3 and angle3 connectionstyles
            import itertools as it
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
                
        fig, ax = plt.subplots(1, 1)
        draw_labeled_multigraph(G, "w", ax)

        fig.tight_layout()
        plt.show()
        # #edges = G.edges()           
        # #edge_widths = [G[u][v]['width'] for u, v in edges]
        # #edge_colours = [G[u][v]['colour'] for u, v in edges]
        # from IPython.display import Image
        # p=nx.drawing.nx_pydot.to_pydot(G)
        # p.write_png('multi.png')
        # Image(filename='multi.png')
           
            
        # # layout setup:
        # # centre_node = 'qubit'  # Or any other node to be in the center
        # # defects_nodes = set(G) - {'qubit'}
        # # pos = nx.circular_layout(G.subgraph(defects_nodes))
        # # pos = nx.circular_layout(G)
        # # pos = nx.spring_layout(G, seed = 0)
        # # pos = nx.multipartite_layout(G, subset_key = 'subset')
        # graphviz_layout = nx.drawing.nx_pydot.graphviz_layout
        # pos = graphviz_layout(G, prog="twopi")
        # #pos = graphviz_layout(G, prog="dot")
        # ##pos = graphviz_layout(G, prog="circo")
        # #pos = graphviz_layout(G, prog="sfdp")
        
        # # render and save graph:
        # plt.figure()
        # plt.title(filename)
        # nx.draw(G,
        #         pos,
        #         #width=edge_widths, 
        #         #edge_color = edge_colours,
        #         node_color = node_colours,
        #         node_size = node_sizes,
        #         labels = node_labels, 
        #         font_size = 12
        #         )
        # import itertools as it
        # connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
        
        # nx.draw_networkx_edge_labels(G,
        #                              pos = pos, 
        #                              edge_labels = edge_labels, 
        #                              font_size = 12)
        # plt.savefig(filename + '.svg')#, dpi=300)#, bbox_inches='tight')

        
