#%%
# CORRELATION
"""
creates corellation matrix from lists of accepted models from one or multiple chains
vectorises models given parameter labels
performs hierarchical clustering
also outputs rearranged correlation matrices given clustering dissimilatiry threshold thresholds
"""

import pandas as pd
import pickle
import configs
import seaborn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np

# settings and source data:
cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '250818-og_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 1
Rs = [1,3,4,5] # [1 + i for i in range(5)]
burns = [100000, 225000, 100000, 100000] # [int(100000) for x in Rs]
hyperparams = configs.get_hyperparams(config_name)
output_name = experiment_name + '_R1,3end,4,5'


# go over all chains to be included:
setup_done = False
for j, R in enumerate(Rs):
    burn = burns[j]
    filename = experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_proposals.pickle'
    with open(filename,
              'rb') as filestream:
        proposals_dict = pickle.load(filestream)
    proposals = proposals_dict['proposals']
    
    print('total acc. prop.: ' + str(len(proposals)))
    
    # accepted proposals after burn-in 
    proposals = proposals[burn:]
    
    print('left unburned: ' + str(len(proposals)))
    
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
        
burn = min(burns) # take minimum for filenames - now can burn more of some chains but this is low bound

        
# correlation matrix inc. plot:        
data = pd.DataFrame(data = parameter_lists, columns = labels[:]) # columns! - now without energies
correlations = data.corr()
plt.figure(figsize=(10,10))
seaborn.heatmap(correlations, annot=False, cmap=cmap, fmt=".2f", linewidths=0.5, vmin = -1, vmax = 1)
# colormaps: coolwarm, PiYG
plt.savefig(output_name + '_burn' + str(burn) + '_correlation_unclustered.svg', dpi = 1000, bbox_inches='tight')


# hierarchical clustering and rearranged correlation matrix:

# dissimilarity matrix, linkage matrix Z, and dendrogram D:
dissimilarity = 1 - abs(correlations)
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



