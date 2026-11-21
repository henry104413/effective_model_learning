#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:35:28 2026

@author: henry
"""



import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import colorcet

# global plotting settings (works locally):
fontsize = 6
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelpad'] = 0.1
plt.rcParams['axes.linewidth'] = 0.25
plt.rcParams['xtick.major.width'] = 0.25
plt.rcParams['ytick.major.width'] = 0.25
single_col_figsize=(3.41,2)
inset_figsize=(1.5,1)
single_col_tall_figsize=(3.41,4)
single_col_square_figsize=(3.41,3.41)
#colours = list(matplotlib.colormaps['Set1'].resampled(15)(range(15)))
colours = colorcet.glasbey_dark[:15]

clustering_name = '261103-4M_std0p05_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_Rs0thru89_e500_clustering'
# clustering_name = '261027-2M-exp_std0_Wit-Fig4-6-0_025_Lsyst-sx,sy,sz-Lvirt-sz,sy,sz-Cs2v-sx,sy,sz-Cv2v-sx,sy,sz-_D2_Rs0thru4_e500_clustering'
SSEs_import = np.genfromtxt(clustering_name + '_SSEs.csv', delimiter=',').transpose()
silh_import = np.genfromtxt(clustering_name + '_silhouette_scores.csv', delimiter=',').transpose()

SSEs = SSEs_import[1]
silh = silh_import[1]
SSEs = SSEs/(a:=np.nanmax(SSEs))
ks = SSEs_import[0]

plt.figure(figsize=inset_figsize)
plt.xlabel(r'$k$')
#plt.ylabel(ops_longlabels[op])
plt.ylim([-0.0, 1])
#plt.plot(ks, SSEs, '-', color = colours[0], linewidth = 1, alpha = 0.75, label = 'normalised ' + r'$SSE$')
#plt.plot(ks, silh, '-', color = colours[1], linewidth = 1, alpha = 0.75, label = r'$silhouette\ score$')
plt.plot(ks, SSEs, '-', color = colours[0], linewidth = 1, alpha = 0.75, label = r'$SSE$')
plt.plot(ks, silh, '-', color = colours[1], linewidth = 1, alpha = 0.75, label = r'$silh.$')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # looks like all integers get tick
#plt.gca().xaxis.get_major_locator().set_params(integer=True) # looks like only some ticks but all are integer
plt.legend(loc = 'upper right')
plt.savefig(clustering_name + '_metrics.svg', dpi = 1000, bbox_inches='tight')
