#%%
# CORRELATION

import pandas as pd
import pickle
import configs
import seaborn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np

cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '250818-sim_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 1
Rs = [1 + i for i in range(5)]
burn = int(100000)
hyperparams = configs.get_hyperparams(config_name)
output_name = experiment_name + '_R1,2,3,4,5'

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
plt.savefig(output_name + '_correlation_unclustered.svg', dpi = 1000, bbox_inches='tight')


# hierarchical clustering and rearranged correlation matrix:

correlations = pdCM
dissimilarity = 1 - abs(correlations)

# linkage matrix and dendrogram:
fig = plt.figure(figsize=(10,5))
ax = fig.gca()

Z = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dissimilarity), 'complete')
D = scipy.cluster.hierarchy.dendrogram(Z, labels=data.columns, orientation='top', 
           leaf_rotation=90, ax=ax)
ax.set_ylabel('dissimilarity')           
plt.savefig(output_name + '_burn' + str(burn) + '_dendrogram.svg', dpi = 1000, bbox_inches='tight')
#%%

# thresholds for clustering to loop over:
thresholds = [0.5, 0.7, 0.9]
for threshold in thresholds:

    # cluster labels and indices to sort them:
    labels = scipy.cluster.hierarchy.fcluster(Z, threshold, criterion='distance')
    labels_order = np.argsort(labels)

    # new dataframe with sorted columns
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
    plt.savefig(output_name + '_burn' + str(burn) + '_thresh' + str(threshold) + '_correlation_clustered.svg', dpi = 1000, bbox_inches='tight')



