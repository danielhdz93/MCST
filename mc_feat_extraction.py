# -*- coding: utf-8 -*-
"""
------------
mc_feat_extraction.py
-------------
Date: November 2023 V0.1

@title: Multi-Channel Signals Tools
@author: Daniel Eduardo Hernández Morales and Leonardo Trujillo
@institution: Tecnológico Nacional de México/Instituto Tecnológico de Tijuana

Module for feature extraction from a single file.
"""

import pandas as pd
import readingM as read


def extract_mc_features(input_file, input_folder, channels, getST, getBP, transposeInput = False, nsiSeg=10):
    """
    Extract EEG features from the input file.
    
    Parameters:
    - input_file (str): The name of the input file.
    - input_folder (str): The path to the input folder.
    - channels (list): List of channels to extract features from. ([X] single channel, [X,Y] dual channel)
    - getST (bool): Whether to extract statistical features.
    - getBP (bool): Whether to extract Band-Power features.
    - transposeInput (bool): If True, transpose the input data, use if channels are row vectors as methods expect them as column vectors.
    
    Returns:
    list: List of extracted EEG features.
    """
    
    if transposeInput:
        pd.read_csv(input_folder+input_file,header=None).transpose().to_csv(input_folder+'temp.csv',header=False, index=False)
        temp = read.reading(input_folder+'temp.csv')
    else:
        temp = read.reading(input_folder+input_file)
    singlest = []
    singlebp = []
    diffst = []
    ratiost = []
    diffbp = []
    ratiobp = []
    
    # Extract features for each channel
    for chan in channels:
        if len(chan) == 2:
            if getST:
                diffst.append(temp.getdiffasymstats(chan[0],chan[1]))
                ratiost.append(temp.getratioasymstats(chan[0],chan[1]))
            if getBP:
                diffbp.append(temp.getdiffasymbp(chan[0],chan[1]))
                ratiobp.append(temp.getratioasymbp(chan[0],chan[1]))
        if len(chan)==1:
            if getST:
                singlest.append(temp.getstatistical(chan[0]))
                singlest.append(temp.gethjorth(chan[0]))
                singlest.append(temp.gethoc(chan[0]))
                singlest.append([temp.getnsi(nsiSeg,chan[0])])
            if getBP:
                singlebp.append(temp.getallfreq(chan[0]))
    
    allf = []
    
    # Combine features
    l = max(len(singlest),len(singlebp))
    for i in range(l):
        if getST and i < len(singlest):
            allf.extend(singlest[i])
        if getBP and i < len(singlebp):
            allf.extend(singlebp[i])
    
    for i in range(len(diffst)):
        allf.extend(diffst[i])
        allf.extend(ratiost[i])
    
    for i in range(len(diffbp)):
        allf.extend(diffbp[i])
        allf.extend(ratiobp[i])
        
    return allf

