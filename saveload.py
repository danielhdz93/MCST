# -*- coding: utf-8 -*-
'''
------------
saveload.py
-------------
Date: November 2023 V0.1

@title: Multi-Channel Signals Tools
@author: Daniel Eduardo Hernández Morales and Leonardo Trujillo
@institution: Tecnológico Nacional de México/Instituto Tecnológico de Tijuana

Methods to read and write the list of features into files
'''
import pickle

# Write
def savefeatures(flist,filename):
    with open(filename, 'wb') as f:
        pickle.dump(flist, f)

# Read
def loadfeatures(filename):
    with open(filename, 'rb') as f:
        my_list = pickle.load(f)
    return my_list
