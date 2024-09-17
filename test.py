#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:38:44 2024

@author: henry
"""

refs = {'sys1' : 43}

test = {'sys1' : (5, 'someguy'), 65 : (3, 'anotherguy')}

print(test)

for key in list(test):
    
    if type(key) == str:
        
        test[refs[key]] = test[key]
        
        del test[key]
        

print(test)