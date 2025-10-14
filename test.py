import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--defects_number', '-D', help = 'number of defects', type = int)

parser.add_argument('--repetition_number', '-R', help = 'repetition number', type = int)

cmdargs = parser.parse_args()

print('Now D = ' + str(cmdargs.defects_number))

print('And R = ' + str(cmdargs.repetition_number))


#%%

import configs.py

# so 
# 1) take model 
# 2) take all possible parameters (energies, couplings in order, Lindblads in order)
# 3) populate that with what's in the model

# parameters vector segments lengths:
# Dx1(single site hamiltonians, currently assumed just sz), 
# (Q x D x len(s2v library)) 
# ((D choose 2) x len(v2v library))
# (Q x len(L syst library))
# (D x len(L virt library))

# so that means:
# spliitings in order of defects
# q2d aka s2v in order of pairs and then ops
# d2d aka v2v in order of pairs and then ops
# qubit/syst Ls in order of qubits and then ops
# defec/virt Ls in order of defects and then ops

# feed hyperparameters dictionary with following entries:
# qubit_Ls_library
# defect_Ls_library
# qubit2defect_couplings_library
# defect2defect_couplings_library




#%%
import configs
import pickle
import matplotlib.pyplot as plt

hyperparameters = configs.get_hyperparams('Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-')

with open('testmodel.pickle', 'rb') as filestream:
    model = pickle.load(filestream)

# to be populated up for each process in the library (vectorised model of fixed length, zeros where processes absent)


def make_op_label(op: str, tex: bool = False)->str:
    """
    For common string names of operators returns a shorthand or latex symbol if tex passed as true.
    """
    match op:
        case 'sigmax':
            if tex: return r'$\sigma_x$'
            else: return 'sx'
        case 'sigmay':
            if tex: return r'$\sigma_y$'
            else: return 'sy'
        case 'sigmaz':
            if tex: return r'$\sigma_z$'
            else: return 'sz'
        case 'sigmap':
            if tex: return r'$\sigma_+$'
            else: return 'sp'
        case 'sigmam':
            if tex: return r'$\sigma_-$'
            else: return 'sm'


# gather all qubits, all defects, qubit-defect pairs, and defect-defect pairs,
# as well as corresponding labels using the system (S) and virtual systems (V) nomenclature

qubits = [x for x in model.TLSs if x.is_qubit]
qubits_labels = ['S' + str(i+1) for i in range(len(qubits))] 

defects = [x for x in model.TLSs if not x.is_qubit]
defects_labels = ['V' + str(i+1) for i in range(len(defects))]
  
q2d_pairs = [(q, d) for q in qubits for d in defects]
q2d_pairs_labels = [qubits_labels[qubits.index(q)] + ',' + defects_labels[defects.index(d)]
                    for q, d in q2d_pairs] 

d2d_pairs = []
for d1 in defects:
    for d2 in defects:
        if d1 != d2 and (d2, d1) not in d2d_pairs:
            d2d_pairs.append((d1, d2))
d2d_pairs_labels = [defects_labels[defects.index(d1)] + ',' + defects_labels[defects.index(d2)]
                    for d1, d2 in d2d_pairs] 


# initialise lists for labels, labels with latex symbols, and parameter values
labels = []
labels_latex = []
values = []
    
# defect splittings:
for defect, label in zip(defects, defects_labels):
    labels.append(label + '-E-')
    #labels_latex.append(label + r'$\ \sigma_z$')
    labels_latex.append(label + r'$\ E$')
    values.append(defect.energy)
    
# q2d in order of pairs and then library operators:
# note: coupling information stored on one of the pair hence check both! 
for pair, pair_label in zip(q2d_pairs, q2d_pairs_labels):
    for coupling in list(hyperparameters['qubit2defect_couplings_library'].keys()):
        # go over all types of coupling in library, check if present on either of this pair with other as partner:
        # note: learning methods should allow it to only exist on one of pair at a time,
        # otherwise it would be duplicate... 
        # but for completenes and assuming linearity can just add together all rates potential instances of that coupling
        
        ops_label = ''
        ops_label_latex = ''
        for i, op_pair in enumerate(coupling): # coupling is TUPLE of tuples!!
            if i == 0: ops_label_latex += ' '
            if i>0: ops_label_latex += '+'
            ops_label += '-' + make_op_label(op_pair[0]) + '-' + make_op_label(op_pair[1])
            ops_label_latex += make_op_label(op_pair[0], tex=True) + r'$\otimes$' + make_op_label(op_pair[1], tex=True)
        
        # labels for this process between these pairs
        labels.append(pair_label + ops_label)
        labels_latex.append(pair_label + ops_label_latex)
        
        # note: couplings stored as:
        # TLS.couplings = {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
        
        # check if exists, then populate rate, otherwise leave as zero:
        value = float(0)
        for TLS1, TLS2 in [pair, tuple(reversed(pair))]:
            # note: takes TLS1 as either of the pair with TLS2 being the other to avoid repetition
            if TLS2 in TLS1.couplings:
                for existing_coupling in TLS1.couplings[TLS2]:
                    # go over all existing ones
                    # if matching one from library ie. "coupling", add its rate to value
                    # existing_coupling is tuple (rate, [(op_self, op_partner), (op_self, op_partner), ...])
                    # coupling is tuple of tuples for operators (from library)
                    if set(existing_coupling[1]) == set(coupling): # ie. type is present
                        value += existing_coupling[0]
        values.append(value)
        
# d2d in order of pairs and then library operators:
# note: coupling information stored on one of the pair hence check both! 
for pair, pair_label in zip(d2d_pairs, d2d_pairs_labels):
    for coupling in list(hyperparameters['defect2defect_couplings_library'].keys()):
        # go over all types of coupling in library, check if present on either of this pair with other as partner:
        # note: learning methods should allow it to only exist on one of pair at a time,
        # otherwise it would be duplicate... 
        # but for completenes and assuming linearity can just add together all rates potential instances of that coupling
        
        ops_label = ''
        ops_label_latex = ''
        for i, op_pair in enumerate(coupling): # coupling is TUPLE of tuples!!
            if i == 0: ops_label_latex += ' '
            if i>0: ops_label_latex += '+'
            ops_label += '-' + make_op_label(op_pair[0]) + '-' + make_op_label(op_pair[1])
            ops_label_latex += make_op_label(op_pair[0], tex=True) + r'$\otimes$' + make_op_label(op_pair[1], tex=True)
        
        # labels for this process between these pairs
        labels.append(pair_label + ops_label)
        labels_latex.append(pair_label + ops_label_latex)
        
        # note: couplings stored as:
        # TLS.couplings = {partner: [(rate, [(op_self, op_partner), (op_self, op_partner), ...]]}
        
        # check if exists, then populate rate, otherwise leave as zero:
        value = float(0)
        for TLS1, TLS2 in [pair, tuple(reversed(pair))]:
            # note: takes TLS1 as either of the pair with TLS2 being the other to avoid repetition
            if TLS2 in TLS1.couplings:
                for existing_coupling in TLS1.couplings[TLS2]:
                    # go over all existing ones
                    # if matching one from library ie. "coupling", add its rate to value
                    # existing_coupling is tuple (rate, [(op_self, op_partner), (op_self, op_partner), ...])
                    # coupling is tuple of tuples for operators (from library)
                    if set(existing_coupling[1]) == set(coupling): # ie. type is present
                        value += existing_coupling[0]
        values.append(value)
 
# qubit Ls in order of qubits and then ops from library:
# note: only assuming one value of each type present
# note: not linear and multiple could exist, but algorithm wouldn't add one if same type exists already
for qubit, qubit_label in zip(qubits, qubits_labels):
    for L in list(hyperparameters['qubit_Ls_library'].keys()):
        labels.append(qubit_label + '-' + make_op_label(L) + '-')
        labels_latex.append(qubit_label + r'$\ $' + make_op_label(L, tex=True))
        if L in qubit.Ls:
            values.append(qubit.Ls[L])
        else:
            values.append(float(0))
        
# defect Ls in order of defects and then ops from library:
# note: only assuming one value of each type present
# ...they are not linear and multiple could exist, but algorithm wouldn't add one if same type exists already
for defect, defect_label in zip(defects, defects_labels):
    for L in list(hyperparameters['defect_Ls_library'].keys()):
        labels.append(defect_label + '-' + make_op_label(L) + '-')
        labels_latex.append(defect_label + r'$\ $' + make_op_label(L, tex=True))
        if L in defect.Ls:
            values.append(defect.Ls[L])
        else:
            values.append(float(0))
    
 
    
plt.figure()
plt.plot(values, range(len(values)), 'b +')
plt.yticks(range(len(values)), labels = labels_latex)#, rotation = 90)
plt.savefig('testvectorisation.svg', dpi = 1000, bbox_inches='tight') 



#%%
# CORRELATION

import pandas as pd
import pickle
import configs
import seaborn
import matplotlib.pyplot as plt

cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '250815-sim_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 2
Rs = [1 + i for i in range(5)]
burn = int(100000)
hyperparams = configs.get_hyperparams(config_name)

setup_done = False

# go over all chains to be
for R in Rs:
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_proposals.pickle'
    with open(filename,
              'rb') as filestream:
        proposals_dict = pickle.load(filestream)
        
    proposals = proposals_dict['proposals']
    
    print(len(proposals))
    
    # clearly all proposals were saved now despite condition on burn-in... ok then
    
    proposals = proposals[burn:]
    
    if not setup_done:
        first_proposal = proposals[0]
        _, labels, labels_latex = first_proposal.vectorise_under_library(hyperparameters = hyperparams)
        labels = labels_latex # currently using the latex labels for labels
        parameter_lists = {name: [] for name in labels}
        setup_done = True
        
    for i, proposal in enumerate(proposals):
    # here proposal is an instance of model with vectorisation method
        vector = proposal.vectorise_under_library(hyperparameters = hyperparams)[0]
        # vectorised in order of labels, hence take n-th entry corresponds to n-th label model parameter
        for j, value in enumerate(vector):
            parameter_lists[labels[j]].append(vector[j])
        
data = pd.DataFrame(data = parameter_lists, columns = labels[:]) # columns! - now without energies

pdCM = data.corr()

plt.figure(figsize=(10,10))
seaborn.heatmap(pdCM, annot=False, cmap=cmap, fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
# colormaps: coolwarm, PiYG
plt.savefig(experiment_name + '_correlation_unclustered.svg', dpi = 1000, bbox_inches='tight')


# try hierarchical clustering:
correlations = pdCM

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import numpy as np

plt.figure(figsize=(10,5))
dissimilarity = 1 - abs(correlations)
Z = linkage(squareform(dissimilarity), 'complete')
dendrogram(Z, labels=data.columns, orientation='top', 
           leaf_rotation=90);
plt.savefig(experiment_name + '_burn' + str(burn) + '_dendrogram.svg', dpi = 1000, bbox_inches='tight')
#%%

# Clusterize the data
threshold = 0.7
labels = fcluster(Z, threshold, criterion='distance')

# Keep the indices to sort labels
labels_order = np.argsort(labels)

# Build a new dataframe with the sorted columns
for idx, i in enumerate(data.columns[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(data[i])
    else:
        df_to_append = pd.DataFrame(data[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)
        
plt.figure(figsize=(10,10))
correlations = clustered.corr()
# seaborn.heatmap(round(correlations,2), cmap='RdBu', annot=True, annot_kws={"size": 7}, vmin=-1, vmax=1);
seaborn.heatmap(correlations, cmap=cmap, annot=False, annot_kws={"size": 7}, vmin=-1, vmax=1);
plt.savefig(experiment_name + '_burn' + str(burn) + '_thresh' + str(threshold) + '_correlation_clustered.svg', dpi = 1000, bbox_inches='tight')



#%%
# CREATE SIMULATED DATA IN OTHER BASES:
    
import numpy as np
import pickle    
import matplotlib.pyplot as plt
    
# measured data wrt sx:
data_file = 'Wit-Fig4-6-0_025.csv'
dataset_no = 0 # means first pair of columns
contents = np.genfromtxt(data_file, delimiter=',')#,dtype=float) 
dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
ts = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
og_sx = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   
order = ts.argsort() # sort by t in case of unsorted:
ts = ts[order]
og_sx = og_sx[order] # original sx measurements - currently using all simulated
ts = ts/1000 # convert from ns to us


# import a model (would have been trained on ts, sx):
model_file = '250810-batch_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R2_best'
with open(model_file + '.pickle', 'rb') as filestream:
    best = pickle.load(filestream)
    

# random noise shift for sy, sz, and simulated measurements of sy, sz:
noise_sy = np.random.normal(loc = 0, scale = 0.01, size = len(ts))
noise_sz = np.random.normal(loc = 0, scale = 0.01, size = len(ts))
noise_sx = np.random.normal(loc = 0, scale = 0.01, size = len(ts))
sy, sz, sx = best.calculate_dynamics(ts, observable_ops = ['sigmay', 'sigmaz', 'sigmax'],
                                        custom_function_on_return = False)
sim_sy = sy + noise_sy
sim_sz = sz + noise_sz
sim_sx = sx + noise_sx

# plot:
plt.figure()
plt.plot(ts, og_sx, 'r.', label = r'og $\langle\sigma_x\rangle}$', alpha = 0.4, markersize = 5)
plt.plot(ts, sx, 'b-', label = r'_$\sigma_x\>$', alpha = 0.5, markersize = 5)
plt.plot(ts, sim_sx, 'b.', label = r'sim $\langle\sigma_x\rangle$', alpha = 0.5, markersize = 5)
plt.plot(ts, sy, 'm-', label = '_hidden', alpha = 0.4)
plt.plot(ts, sim_sy, 'm.', label = r'sim $\langle\sigma_y\rangle$', alpha = 0.5, markersize = 5)
plt.plot(ts, sz, 'g-', label = '_hidden', alpha = 0.4)
plt.plot(ts, sim_sz, 'g.', label = r'sim $\langle\sigma_z\rangle$', alpha = 0.5, markersize = 5)
plt.legend(fontsize = 14)
plt.xlabel(r'$time\ (\mu s)$') 
plt.savefig('simulated_' + model_file + '.svg', dpi = 1000, bbox_inches='tight')

simulated_data = {'ts': ts,
                  'sx': sim_sx,
                  'sy': sim_sy,
                  'sz': sim_sz
                  }

with open('simulated_' + model_file + '.pickle', 'wb') as filestream:
    pickle.dump(simulated_data, filestream)


#%%
# bar plots of champion loss for different configurations (libraries)
if not True: 
    
    plt.rcParams["font.size"] = 16

    configs = ['Lsyst-sz-Lvirt--Cs2v-sx-Cv2v--', 
               'Lsyst-sz-Lvirt--Cs2v-sx-Cv2v-sx-', 
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v--', 
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx-Cv2v-sx-',
               'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx-Cv2v--',
               'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx-Cv2v-sx-',
               'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-',
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v--',
               'Lsyst-sx,sy,sz-Lvirt--Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-',
               'Lsyst-sx,sy,sz-Lvirt-sx,sy,sz-Cs2v-sx,sy,sz-Cv2v--', 
               'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-',
               'Lsyst-sx-Lvirt--Cs2v-sx-Cv2v--',
               'Lsyst-sx-Lvirt--Cs2v-sz-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sz-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sx,sy-Cv2v--',
               'Lsyst-sz-Lvirt--Cs2v-sx,sz-Cv2v--'
               ]
    configs = [configs[x] for x in [0, 15, 16, 6, 7, 11]]
    
    experiment_base = '250621'
    target_file = 'Wit-Fig4-6-0_025'
    losses = {'sx': [], 'full': []}
    labels = {'sx': [], 'full': []}
    for config in configs: 
        for regime in ['sx', 'full']:
            experiment_name = regime + '_' + experiment_base + '_' + target_file + '_' + config 
            for D in [2]: # tuple of Ds   
                losses[regime].append(return_champion_loss(experiment_name, D))
                labels[regime].append(config)
                
    def ops_label_dresser(ops: str, setname: str) -> str: # works
        """
        Argument is comma-delimiting string containing sx, sy, sz.
        
        Returns dressed string of operators to show in as processes of each class,
        assuming setname containing "C" means it's symmetric coupling (ie. applied on both subsystems).
        """
        ops_list = [x for x in ops.split(',') if x] # separates by comma and eliminates empty strings
        symbols = {'sx': r'$\sigma_x$',
                   'sy': r'$\sigma_y$',
                   'sz': r'$\sigma_z$'}
        output = ''
        for x in ops_list:
            if output: output = output + ', '
            output = output + symbols[x]
            if 'C' in setname: output = output + r'$\otimes$' + symbols[x] 
        return output
        
    def interpret_config(string: str) -> str:
        """
        Argument is configuration label string of form Lsyst-ops-Lvirt-ops-Cs2v-ops-Cv2v-ops-,
        where ops is comma-delimited string containg any of sx, sy, sz.
        
        Returns dressed label of Ls and couplings as sets of operators;
        each single operator taken as acting on both systems if coupling.
        """
        Lsyst_Lvirt_Cs2v_Cv2v = [string.split('-')[x] for x in (1,3,5,7)]
        setnames = [r'$L_{syst}$', r'$L_{virt}$', r'$C_{s2v}$', r'$C_{v2v}$']
        label = ''
        for i, ops in enumerate(Lsyst_Lvirt_Cs2v_Cv2v):
            if i>0: label = label + '\n'
            label = label + setnames[i] + r'$\in\{$' + ops_label_dresser(ops, setnames[i]) + r'$\}$'
        
        return label    
            
    # A = ([interpret_config(x) for x in labels])
            
        
        
        
  #%%  
    #losses = [x/max(losses) for x in losses]
    # just for sorting (I think):
    # losses_arr = np.array(losses)
    # order = losses_arr.argsort()
    # labels_arr = np.array([interpret_config(x) for x in labels])
    # losses_arr_sorted = losses_arr[order]
    # labels_arr_sorted = labels_arr[order]
    # colours = np.array(colours)[order]
    # labels_sorted = [labels[i] for i in order]
    
    plt.figure(figsize=(10, 30))
    plt.barh(np.arange(len(losses['sx'])), losses['sx'], color='r', alpha=0.5)
    plt.barh(np.arange(len(losses['full'])), losses['full'], color='b', alpha=0.5)
    #plt.xscale('log')
    plt.yticks(ticks = np.arange(len(losses['sx'])), labels = label['sx'])
    plt.title('D = 2')
    plt.xlabel('lowest loss')#
    ax=plt.gca()
    #ax.axis["left"].major_ticklabels.set_ha("left")
    #ax.set_xticklabels(ha='left')
    plt.savefig('250621_loss_vs_configuration_full_vs_sx' + '.svg', dpi = 1000, bbox_inches='tight')
  

#%%
import pickle
with open('/home/henry/Code/effective_model_learning/250822-sim-small_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R1_proposals.pickle', 'rb') as filestream:
    A = pickle.load(filestream)
    
#%% 
# plot selected different proportion learning on top of target data
# very ad hoc

import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams["font.size"] = 16    
dataset_no = 0

if True:    
    
    
    # target data:
    contents = np.genfromtxt('Wit-Fig4-6-0_025.csv',delimiter=',')#,dtype=float) 
    dataset = contents[:,[2*dataset_no, 2*dataset_no + 1]]
    xs = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 0]     
    ys = dataset[np.isfinite(dataset[:,0]) + np.isfinite(dataset[:,1]), 1]   

    # sort by x and rescale to us:
    order = xs.argsort()
    xs = xs[order]
    ts = xs/1000
    ys = ys[order]
    
    # plot setup:
    plt.figure()
    plt.ylabel(r'<$\sigma_x$>')
    plt.xlabel(r'time ($\mu s$)')
    plt.ylim([-0.5, 1.05])
    plt.xlim([-0.05, 2.55])
    
    # colours and labels for models in order:
    colours = (x for x in ['green' for y in range(3)]
                         +['purple' for y in range(3)]
                         +['red' for y in range(3)])
    
    alphas = (x for x in [0.3 for y in range(3)]
                        +[0.5 for y in range(3)]
                        +[0.7 for y in range(3)])
    
    labels = (x for x in ['prediction', '_', '_',
                           'prediction', '_', '_',
                           'prediction', '_', '_'])
    styles = (x for x in ['--', '--', '--',
                          '--', '--', '--',
                          ':', ':', ':'])
    cutoffs = (x for x in [([0.625, 0.625],[-0.5,1]) for y in range(1)]
                         +[([1.25, 1.25],[-0.5,1]) for y in range(1)]
                         +[([1.875, 1.875],[-0.5,1]) for y in range(1)])
    
    
    # import model files, calculate their dynamics: 
    model_files = [# 0.25 proportion:
                   '250825-pred-frac0_25_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R1_best.pickle',
                   '250825-pred-frac0_25_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R5_best.pickle',
                   '250825-pred-frac0_25_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R4_best.pickle',
                   # 0.5 proportion:
                   '250825-pred-frac0_5_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R1_best.pickle',
                   '250825-pred-frac0_5_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R3_best.pickle',
                   '250825-pred-frac0_5_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R5_best.pickle',
                   # 0.75 proportion:
                   '250825-pred-frac0_75_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R5_best.pickle',
                   '250825-pred-frac0_75_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R4_best.pickle',
                   '250825-pred-frac0_75_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_R3_best.pickle'
                   ]
    models = []
    for model_file in model_files:
        with open(model_file, 'rb') as filestream:
            models.append(pickle.load(filestream))
    evolutions = [model.calculate_dynamics(ts, observable_ops = ['sigmax'])[0] for model in models]
    evolutions_gen = (x for x in evolutions)

    # plot predicted dynamics, target data, and save:

    for x in range(3):
        plt.figure()
        plt.ylabel(r'<$\sigma_x$>')
        plt.xlabel(r'time ($\mu s$)')
        plt.ylim([-0.5, 1.05])
        plt.xlim([-0.05, 2.55])
        plt.plot(ts, ys, 'b.', label = 'target',  markersize = 10, markeredgewidth = 0.5, markerfacecolor='None', alpha = 0.5)
        plt.plot(*next(cutoffs), '-', c='orange', label='training cutoff')
        for y in range(3):
            plt.plot(ts, next(evolutions_gen), next(styles), c = next(colours)
                    ,label = next(labels)
                    ,alpha = next(alphas), 
                    )
            
        # plot target data:
        legend = plt.legend()#title = r'$D=' + str(candidate_set.defects_number) + '$' + ', ' + varying)
        plt.savefig('250525_proportion_comparison' + str(x)
                    +'.svg', dpi = 1000, bbox_inches='tight')
        
 


#%%
# output loss of just accepted models (plot and pickle of list)

import pickle
import configs
import matplotlib.pyplot as plt
import numpy as np

cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '250818-sim_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 1
Rs = [1,2,3,4,5]
burn = 0 #int(100000)
hyperparams = configs.get_hyperparams(config_name)

setup_done = False

# go over all chains to be included:
for R in Rs:
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_proposals.pickle'
    with open(filename,
              'rb') as filestream:
        proposals_dict = pickle.load(filestream)
        
    proposals = proposals_dict['proposals']
    
    print(len(proposals))
    
    # accepted proposals after "burn" 
    # proposals = proposals[burn:]
    
    if not setup_done:
        first_proposal = proposals[0]
        _, labels, labels_latex = first_proposal.vectorise_under_library(hyperparameters = hyperparams)
        labels = labels_latex # currently using the latex labels for labels
        parameter_lists = {name: [] for name in labels}
        setup_done = True
        
    for i, proposal in enumerate(proposals):
    # here proposal is an instance of model with vectorisation method
        vector = proposal.vectorise_under_library(hyperparameters = hyperparams)[0]
        # vectorised in order of labels, hence take n-th entry corresponds to n-th label model parameter
        for j, value in enumerate(vector):
            parameter_lists[labels[j]].append(vector[j])
        
    accepted_loss = [x for (x, y) in zip(proposals_dict['loss'][1:], proposals_dict['acceptance']) if y]
    best_loss = min(accepted_loss)
    
    plt.figure()
    plt.plot(accepted_loss, ' .', c = 'orange', linewidth = 0.3, markersize = 0.1)
    plt.yscale('log')
    plt.xlabel('accepted proposal no.')
    plt.ylabel('loss')
    plt.text(0, #(plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0])/20,
             10**(0.98*np.log10(plt.gca().get_ylim()[0])),
             'best loss = ' + '{:.2e}'.format(best_loss))
    plt.savefig(experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_accepted_loss.svg', dpi = 1000, bbox_inches='tight')
    
    with open(experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_accepted_loss.pickle', 'wb') as filestream:
        pickle.dump(accepted_loss, filestream)
        
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn
datafile = '250818-sim-1T-4JL-2tweak_Wit-Fig4-6-0_025_clust_regions5_clustering_assignments.csv'
A = np.genfromtxt(datafile,delimiter=',', skip_header=1)

# fig = plt.figure()
# #plot = plt.imshow(A, cmap='hot', interpolation='nearest')
# #fig.colorbar(plot, orientation='horizontal', fraction=.1)
# plt.show()
# cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# seaborn.heatmap(A, annot=False, cmap=cmap, fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
# plt.ylabel('model')
# plt.xlabel('cluster')
# # colormaps: coolwarm, PiYG
# plt.savefig(datafile[:-4] + '.svg', dpi = 1000, bbox_inches='tight')
# plt.show()

# paste bounds here:
bounds = [(40000, 41500),
          (52000, 56000),
          (62000, 64000),
          (74000, 75000),
          (83000, 85000)
          ]
window_widths = [y - x for (x, y) in bounds]
window_edges = [sum(window_widths[:i]) for i in range(len(window_widths))]

for k in [2, 3, 4, 5]:
    B = A[k-2,:]
    fig = plt.figure()
    ax = fig.gca()
    ax.set_yticks(list(set(B)))
    plt.plot(range(len(B)), B, linewidth = 0.5, c = 'teal')
    plt.title(r'$k=$' + str(k))
    plt.xlabel('model label')
    plt.ylabel('cluster label')
    for edge in window_edges:
        plt.plot([edge, edge], [min(B), max(B)], '--', c = 'orange')
    plt.savefig(datafile[:-4] + '_k' + str(k) + '.svg', dpi = 1000, bbox_inches='tight')

#%%

import numpy as np
import matplotlib.pyplot as plt

Ns = [100, 500, 1000, 5000, 10000]
min_clusters = 2
max_clusters = 10
clusters_counts = list(range(min_clusters, max_clusters + 1))
filename = '250818-sim-1T-4JL-2tweak_Wit-Fig4-6-0_025_clustering_profiling_test4clustering_profiling'
times = np.genfromtxt(filename + '.txt', delimiter = ',')
avgs_each_N = np.mean(times, axis=0)

plt.figure(tight_layout = True)
plt.matshow(times)
plt.xlabel(r'$N$')
plt.ylabel(r'$k$')
plt.xticks(ticks = range(len(Ns)), labels = [str(x) for x in Ns])
plt.yticks(ticks = range(len(clusters_counts)), labels = clusters_counts)
plt.colorbar(fraction = 0.1, label = r'$time\ \mathrm{(s)}$')
plt.savefig(filename + '_profiling_checkerboard.svg', bbox_inches='tight')

plt.figure(tight_layout = True)
plt.xlabel(r'$N$')
plt.ylabel(r'$\sqrt{time\ \mathrm{(s)}}$')
Y = np.sqrt(avgs_each_N)
plt.plot(Ns, Y, ':+', linewidth = 1, markeredgewidth = 2, c='firebrick')
plt.savefig(filename + '_profiling_t_vs_N_plot.svg', bbox_inches='tight')