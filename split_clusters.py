#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effective model learning
@author: Henry (henry104413)
"""

import sys # for passing command line arguments
import pickle
import json




try:
    experiment_name = str(sys.argv[1])
except:
    print('Clustering:\n Using default experiment name in filename'
          +'\n - not command line argument')
    experiment_name = ''
try:
    defects_number = str(sys.argv[2])
except:
    print('Clustering:\n Using default defects number in filename'
          +'\n - not command line argument')
    defects_number = 1


clustering_output_file = 