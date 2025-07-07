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
hyperparameters = configs.get_hyperparams('Lsyst-sz-Lvirt--Cs2v-sx,sz-Cv2v--')

with open('testmodel.pickle', 'rb') as filestream:
    model = pickle.load(filestream)

def op_label(op: str, tex: bool = False)->str:
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

qubits = [x for x in model.TLSs if x.is_qubit]
qubits_count = len(qubits)
qubits_labels = ['S' + str(i+1) for i in range(len(qubits))] 

defects = [x for x in model.TLSs if not x.is_qubit]
defects_count = len(defects)
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



# qubit_Ls_library = hyperparameters['qubit_Ls_library']
# defect_Ls_library = hyperparameters['defect_Ls_library']
# qubit2defect_couplings_library = hyperparameters['qubit2defect_couplings_library']
# defect2defect_couplings_library = hyperparameters['defect2defect_couplings_library']

process_class_lib_names = ['qubit2defect_couplings_library',
                           'defect2defect_couplings_library',
                           'qubit_Ls_library',
                           'defect_Ls_library']
                          
process_class_libs = [list(hyperparameters[key].keys()) for key in process_class_lib_names]

# targets = defects + q2d_pairs + d2d_pairs + qubits + defects

# initialise lists for labels, labels with latex symbols, and parameter values
# to be populated up to length of total vectorised model
labels = [] #[d + '-sz-' for d in defects_labels]
labels_latex = [] #[d + r'$\sigma_z$' for d in defects_labels]
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
        
        PROBLEM BELOW:
        COUPLING IS NOT A TUPLE, ITS A TUPLE OF TUPLES!! SO LABELS FAIL
        JUST ITERATE OVER ALL ELEMENTS OF TUPLE AND ADD TOGETHER... 
        
        ops_label = ''
        ops_label_latex = ''
        for i, op_pair in enumerate(coupling): # coupling is TUPLE of tuples!!
            if i>0: ops_label_latex += ' +'
            ops_label += '-' + op_label(coupling[0]) + '-' + op_label(coupling[1])
            ops_label_latex += ' ' + op_label(coupling[0], tex=True) + r'$\otimes$' + op_label(coupling[1], tex=True)
        
        # labels for this process between these pairs
        labels.append(pair_label + '-' + op_label(coupling[0]) + '-' + op_label(coupling[1]))
        labels_latex.append(pair_label + ' ' + op_label(coupling[0], tex=True) + r'$\otimes$' + op_label(coupling[1], tex=True))
        
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
for pair, label in zip(d2d_pairs, d2d_pairs_labels):
    for coupling in list(hyperparameters['defect2defect_couplings_library'].keys()):
        # go over all types of coupling in library, check if present on either of this pair with other as partner:
        # note: learning methods should allow it to only exist on one of pair at a time,
        # otherwise it would be duplicate... 
        # but for completenes and assuming linearity can just add together all rates potential instances of that coupling
        
        # labels for this process between these pairs
        labels.append(label + '-' + op_label(coupling[0]) + '-' + op_label(coupling[1]))
        labels_latex.append(label + ' ' + op_label(coupling[0], tex=True) + r'$\otimes$' + op_label(coupling[1], tex=True))
        
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
        



