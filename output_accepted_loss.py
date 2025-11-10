#%%
# output loss of just accepted models (plot and pickle of list)

import pickle
import configs
import matplotlib.pyplot as plt
import numpy as np

cmap = 'RdBu' # 'RdBu' or 'PiYG' are good
# experiment_name = '250811-sim-250810-batch-R2-plus_Wit-Fig4-6-0_025'
experiment_name = '250818-sim-1T-4JL-2tweak_Wit-Fig4-6-0_025' # including experiment base and source file name
config_name = 'Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-'
D = 2
Rs = [2]#,3,4,5,6,7]
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
        
    # plot loss:
    if True:
        accepted_loss = [x for (x, y) in zip(proposals_dict['loss'][1:], proposals_dict['acceptance']) if y]
        best_loss = min(accepted_loss)
        plt.figure()
        plt.plot(accepted_loss, '-', c = 'orange', linewidth = 0.3, markersize = 0.1)
        plt.yscale('log')
        plt.xlabel('accepted proposal no.')
        plt.ylabel('loss')
        plt.text(0, #(plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0])/20,
                 10**(0.98*np.log10(plt.gca().get_ylim()[0])),
                 'best loss = ' + '{:.2e}'.format(best_loss))
        plt.savefig(experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_accepted_loss.svg', dpi = 1000, bbox_inches='tight')
        with open(experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_accepted_loss.pickle', 'wb') as filestream:
                  pickle.dump(accepted_loss, filestream)
   
    # extract and save accepted proposals:    
    if True:
        accepted_proposals = proposals_dict['proposals'] # only accepted ones are stored!
        with open(experiment_name + '_' + config_name + '_D' + str(D) + '_R' + str(R) + '_accepted_proposals.pickle', 'wb') as filestream:
                  pickle.dump(accepted_proposals, filestream)
                  
    # extract sub threshold models and show loss on top of overall loss!
    # to use for analysing SECTIONS of chain!!!
            
