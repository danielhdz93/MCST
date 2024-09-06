# -*- coding: utf-8 -*-
"""
------------
simple extraction sample.py
-------------
Date: November 2023 V0.1

@title: Multi-Channel Signals Tools
@author: Daniel Eduardo Hernández Morales and Leonardo Trujillo
@institution: Tecnológico Nacional de México/Instituto Tecnológico de Tijuana

Sample feature extraction from a multi-channel signal 
"""

import MultiChannelSignalsTools as mcst

input_folder = './'
out_folder = './'
sample_file_name = "sample_data.csv"

#Flags to define what features to extract ST (statistical), BP (frequency)
getST = False 
getBP = True

#List of channels to extract features
#[14], [22] single channel features will be extracted using the data found in each channel
#[14,22] data from these two channels will be used to extract assymetrical features (differencial and rational)
channels = [[14]] #, [22], [14,22]]

#list of extracted features from all the channles defined by the user
feats = mcst.extract_mc_features(sample_file_name, input_folder, channels, getST, getBP, True)

import pandas as pd
data=pd.Series(feats)
data.plot()

