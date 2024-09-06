# -*- coding: utf-8 -*-
"""
------------
split_csv.py
-------------
Date: November 2023 V0.1

@author: Daniel Eduardo Hernández Morales and Leonardo Trujillo
@institution: Tecnológico Nacional de México/Instituto Tecnológico de Tijuana
"""
import pandas as pd
import os


def split_csv(input_file, input_folder, freq, time_window, headers):
    if headers:
        data = pd.read_csv(input_folder+input_file)
    else:
        data = pd.read_csv(input_folder+input_file, header=None)
    
    i=1
    for i in range(int(data.columns.size/(freq*time_window))):
        tmp = data[range(i*(freq*time_window),(i+1)*(freq*time_window))]
        
        if not os.path.exists(input_folder+'split/'):
            os.makedirs(input_folder+'split/')
            
        tmp.to_csv(input_folder+'split/'+input_file[:-4]+
                   '_'+str(i*(freq*time_window))+'-'+str((i+1)*(freq*time_window))+'.csv'
                   , index=False, header=False)
