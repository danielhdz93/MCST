# -*- coding: utf-8 -*-
"""
------------
readingM.py
-------------
Date: November 2023 V0.1

@title: Multi-Channel Signals Tools
@author: Daniel Eduardo Hernández Morales and Leonardo Trujillo
@institution: Tecnológico Nacional de México/Instituto Tecnológico de Tijuana
Class for loading an multi-channel signal into a python object. Additionally, 
it contains the methods for feature extraction for single and dual channel features,
in the time, frequency and time-frequency domains.
"""

import math
import numpy as np
from scipy.fftpack import fft
import pywt

class reading:
    """
    A class for loading and processing multi-channel signals.
    
    Attributes:
        path (str): The path to the file containing the signal data.
        data (list): A list containing channels of the loaded signal.
        hoc, hjorth, statistical, nsi, normdata, all, bp_delta, bp_theta, bp_salfa,
        bp_alfa, bp_beta, bp_gamma, diffentro, ratio_ba, allfreq, allbp, delta, theta,
        salfa, alfa, beta, gamma, rms, ree, alltimefreq, dermean, allfeatures (list): 
        Lists to store various computed features for each channel.
        samples (int): The number of samples in the loaded signal.
        derdata, derderdata, fdata, freqs (list): Lists to store derived data and
        frequency information for each channel.
    """

    def __init__(self,path_to_file):
        """
        Constructor for the reading class.
        
        Args:
            path_to_file (str): The path to the file containing the signal data.
        """    
        
        self.path = path_to_file
        
        with open(path_to_file) as f:
            datatemp  = [x.strip() for x in f.readlines()]
            
        coun = len(datatemp[0].split(','))
        
        self.data = []
        for i in range(0,coun):
            lista = []
            self.data.append(lista)
        
        for line in datatemp:
            elecs = line.split(',')
            for i in range(0,coun):
                self.data[i].append(elecs[i])
        
        self.hoc = []
        self.hjorth = []
        self.statistical = []
        self.nsi = []
        self.normdata = []
        self.all = []
        self.bp_delta = []
        self.bp_tetha = []
        self.bp_salfa = []
        self.bp_alfa = []
        self.bp_beta = []
        self.bp_gamma = []
        self.diffentro = []
        self.ratio_ba = []
        self.allfreq = []
        self.allbp = []
        self.delta = []
        self.tetha = []
        self.salfa = []
        self.alfa = []
        self.beta = []
        self.gamma = []
        self.rms = []
        self.ree = []
        self.alltimefreq = []
        self.dermean = []
        self.allfeatures = []
        
        for i in range(0,coun):
            self.hoc.append(None)
            self.dermean.append(None)
            self.hjorth.append([None]*3)
            self.statistical.append([None]*7)
            self.nsi.append(None)
            self.normdata.append(None)
            self.all.append(None)
            self.allfeatures.append(None)
            self.bp_delta.append([None]*4)
            self.bp_tetha.append([None]*4)
            self.bp_salfa.append([None]*4)
            self.bp_alfa.append([None]*4)
            self.bp_beta.append([None]*4)
            self.bp_gamma.append([None]*4)
            self.diffentro.append([None]*6)
            self.ratio_ba.append(None)
            self.allfreq.append(None)
            self.allbp.append(None)
            self.delta.append(None)
            self.tetha.append(None)
            self.salfa.append(None)
            self.alfa.append(None)
            self.beta.append(None)
            self.gamma.append(None)
            self.rms.append(None)
            self.ree.append([None]*5)
            self.alltimefreq.append(None)
        
        self.samples = len(self.data[0])
        self.derdata = []
        self.derderdata = []
        self.fdata = []
        self.freqs = []
        
        for i in range(0,coun):
            self.derdata.append(None)
            self.derderdata.append(None)
            self.fdata.append(None)
            self.freqs.append(None)
        
        
###############################################################################

    def getall(self, num, coun):
        """
        Computes and returns a list with all time-domain features for a specified channel.
        
        Args:
            num (int): The number of segments for the NSI feature.
            coun (int): The channel number for which features are computed.
        
        Returns:
            list: A list containing all time-domain features for the specified channel.
        
        Note:
            The computed features include statistical features, Hjorth parameters, NSI, and HOC.
        
        Example:
            To get all time-domain features for channel 0:
            ```python
            features = instance_of_reading.getall(0, total_channels)
            ```
        """
        # Method implementation
        a = self.getstatistical(coun)
        b = self.gethjorth(coun)
        c = self.getnsi(num,coun)
        d = self.gethoc(coun)
        all = []
        all.extend(a)
        all.extend(b)
        all.append(c)
        all.extend(d)
        self.all[coun] = all
        self.labelstime = ['power','mean','std','1st diff','2nd diff','normal 1st diff','normal 2nd diff','Activity','Mobility','Complexity','NSI']
        for i in range (0,len(d)):
            self.labelstime.append('HOC #'+str(i))
        return self.all[coun]
     
    def getderdata(self, coun):
        """
        Computes and returns the first derivative of the data for a specified channel.

        Args:
            coun (int): The channel number for which the first derivative is computed.

        Returns:
            list: A list containing the first derivative of the data for the specified channel.

        Example:
            To get the first derivative for channel 0:
            ```python
            derivative_data = instance_of_reading.getderdata(0)
            ```
        """
        # Method implementation
        der = []
        for index in range(0,len(self.data[coun])-2):
            a = (float(self.data[coun][index+2])-float(self.data[coun][index]))/2
            der.append(a)
        return der
    
    def getderderdata(self, coun):
        """
        Computes and returns the second derivative of the data for a specified channel.

        Args:
            coun (int): The channel number for which the second derivative is computed.

        Returns:
            list: A list containing the second derivative of the data for the specified channel.

        Example:
            To get the second derivative for channel 0:
            ```python
            second_derivative_data = instance_of_reading.getderderdata(0)
            ```
        """
        # Method implementation
        der = []
        for index in range(0,len(self.derdata[coun])-2):
            a = (float(self.derdata[coun][index+2])-float(self.derdata[coun][index]))/2
            der.append(a)
        return der
   
    def getstatistical(self, coun):
        """
        Computes and returns a list of statistical features for the data of a specified channel.

        Args:
            coun (int): The channel number for which statistical features are computed.

        Returns:
            list: A list containing statistical features, including power, mean, standard deviation,
            1st and 2nd differences, normalized 1st and 2nd differences for the specified channel.

        Example:
            To get statistical features for channel 0:
            ```python
            statistical_features = instance_of_reading.getstatistical(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][0] == None):
            self.statistical[coun][0] = self.calculate_power(coun)
        if (self.statistical[coun][1] == None):
            self.statistical[coun][1] = self.calculate_mean(coun)
        if (self.statistical[coun][2] == None):
            self.statistical[coun][2] = self.calculate_standarddevation(coun)
        if (self.statistical[coun][3] == None):
            self.statistical[coun][3] = self.calculate_1stdiff(coun)
        if (self.statistical[coun][4] == None):
            self.statistical[coun][4] = self.calculate_2nddiff(coun)
        if (self.statistical[coun][5] == None):
            self.statistical[coun][5] = self.calculate_normal1stdiff(coun)
        if (self.statistical[coun][6] == None):
            self.statistical[coun][6] = self.calculate_normal2nddiff(coun)
        return self.statistical[coun]
    
    def getpower(self,coun):
        """
        Computes and returns the power of the data for a specified channel.

        Args:
            coun (int): The channel number for which the power is computed.

        Returns:
            float: The power of the specified channel.

        Example:
            To get the power of channel 0:
            ```python
            power_value = instance_of_reading.getpower(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][0] == None):
            self.statistical[coun][0] = self.calculate_power(coun)
        return self.statistical[coun][0]

    def calculate_power(self, coun):
        """
        Computes the power of the data for a specified channel and stores it.

        Args:
            coun (int): The channel number for which the power is computed.

        Returns:
            float: The power of the specified channel.

        Example:
            To get the power of channel 0:
            ```python
            power_value = instance_of_reading.calculate_power(0)
            ```
        """
        # Method implementation
        sum = 0
        for index in self.data[coun]:
            sum = sum + pow(float(index),2)
        power = sum/len(self.data[coun])
        return power

    def getmean(self, coun):
        """
        Computes and returns the mean of the data for a specified channel.

        Args:
            coun (int): The channel number for which the mean is computed.

        Returns:
            float: The mean of the specified channel.

        Example:
            To get the mean of channel 0:
            ```python
            mean_value = instance_of_reading.getmean(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][1] == None):
            self.statistical[coun][1] = self.calculate_mean(coun)
        return self.statistical[coun][1]

    def calculate_mean(self,coun):
        """
        Computes and stores the mean of the data for a specified channel.

        Args:
            coun (int): The channel number for which the mean is computed.

        Returns:
            float: The mean of the specified channel.

        Example:
            To get the mean of channel 0:
            ```python
            mean_value = instance_of_reading.calculate_mean(0)
            ```
        """
        # Method implementation
        sum = 0
        for index in self.data[coun]:
            sum = sum + float(index)
        mean = sum/len(self.data[coun])
        return mean

    
    def getstandarddevation(self,coun):
        """
        Retrieves or calculates and returns the standard deviation of the data for a specified channel.

        Args:
            coun (int): The channel number for which the standard deviation is obtained.

        Returns:
            float: The standard deviation of the specified channel.

        Example:
            To get the standard deviation of channel 0:
            ```python
            std_deviation = instance_of_reading.getstandarddevation(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][2] == None):
            if (self.statistical[coun][1] == None):
                self.statistical[coun][1] = self.calculate_mean(coun)
            self.statistical[coun][2] = self.calculate_standarddevation(coun)
        return self.statistical[coun][2]

    def calculate_standarddevation(self, coun):
        """
        Calculates and stores the standard deviation of the data for a specified channel.

        Args:
            coun (int): The channel number for which the standard deviation is obtained.

        Returns:
            float: The standard deviation of the specified channel.

        Example:
            To get the standard deviation of channel 0:
            ```python
            std_deviation = instance_of_reading.calculate_standarddevation(0)
            ```
        """
        # Method implementation
        sum = 0
        for index in self.data[coun]:
            sum = sum + pow((float(index)-self.statistical[coun][1]),2)
        sd = math.sqrt(sum/len(self.data[coun]))
        return sd

    def get1stdiff(self,coun):
        """
        Retrieves or calculates and returns the first difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the first difference is obtained.

        Returns:
            list: A list containing the first difference values of the specified channel.

        Example:
            To get the first difference of channel 0:
            ```python
            first_diff = instance_of_reading.get1stdiff(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][3] == None):
            self.statistical[coun][3] = self.calculate_1stdiff(coun)
        return self.statistical[coun][3]

    def calculate_1stdiff(self, coun):
        """
        Calculates and stores the first difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the first difference is obtained.

        Returns:
            list: A list containing the first difference values of the specified channel.

        Example:
            To get the first difference of channel 0:
            ```python
            first_diff = instance_of_reading.calculate_1stdiff(0)
            ```
        """
        # Method implementation
        sum = 0
        for i in range(0,(len(self.data[coun])-1)):
            sum = sum + abs(float(self.data[coun][i+1])-float(self.data[coun][i]))
        diff1 = sum/(len(self.data[coun]) - 1)
        return diff1

    def get2nddiff(self, coun):
        """
        Retrieves or calculates and returns the second difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the second difference is obtained.

        Returns:
            list: A list containing the second difference values of the specified channel.

        Example:
            To get the second difference of channel 0:
            ```python
            second_diff = instance_of_reading.get2nddiff(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][5] == None):
            self.statistical[coun][5] = self.calculate_2nddiff(coun)
        return self.statistical[coun][5]

    def calculate_2nddiff(self, coun):
        """
        Calculates and stores the second difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the second difference is obtained.

        Returns:
            list: A list containing the second difference values of the specified channel.

        Example:
            To get the second difference of channel 0:
            ```python
            second_diff = instance_of_reading.calculate_2nddiff(0)
            ```
        """
        # Method implementation
        sum = 0
        for i in range(0,(len(self.data[coun])-2)):
            sum = sum + abs(float(self.data[coun][i+2])-float(self.data[coun][i]))
        diff2 = sum/(len(self.data[coun]) - 2)
        return diff2

    def getnormal1stdiff(self, coun):
        """
        Retrieves or calculates and returns the normalized first difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the normalized first difference is obtained.

        Returns:
            list: A list containing the normalized first difference values of the specified channel.

        Example:
            To get the normalized first difference of channel 0:
            ```python
            normal_1st_diff = instance_of_reading.getnormal1stdiff(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][4] == None):
            if (self.statistical[coun][3] == None):
                self.statistical[coun][3] = self.calculate_1stdiff(coun)
            if (self.statistical[coun][1] == None):
                self.statistical[coun][1] = self.calculate_mean(coun)
            if (self.statistical[coun][2] == None):
                self.statistical[coun][2] = self.calculate_standarddevation(coun)
            self.statistical[coun][4] = self.calculate_normal1stdiff(coun)
        return self.statistical[coun][4]

    def calculate_normal1stdiff(self, coun):
        """
        Calculates and stores the normalized first difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the normalized first difference is obtained.

        Returns:
            list: A list containing the normalized first difference values of the specified channel.

        Example:
            To get the normalized first difference of channel 0:
            ```python
            normal_1st_diff = instance_of_reading.calculate_normal1stdiff(0)
            ```
        """
        # Method implementation
        ndiff1 = self.statistical[coun][3]/self.statistical[coun][2]
        return ndiff1

    def getnormal2nddiff(self, coun):
        """
        Retrieves or calculates and returns the normalized second difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the normalized second difference is obtained.

        Returns:
            list: A list containing the normalized second difference values of the specified channel.

        Example:
            To get the normalized second difference of channel 0:
            ```python
            normal_2nd_diff = instance_of_reading.getnormal2nddiff(0)
            ```
        """
        # Method implementation
        if (self.statistical[coun][6] == None):
            if (self.statistical[coun][5] == None):
                self.statistical[coun][5] = self.calculate_2nddiff(coun)
            if (self.statistical[coun][1] == None):
                self.statistical[coun][1] = self.calculate_mean(coun)
            if (self.statistical[coun][2] == None):
                self.statistical[coun][2] = self.calculate_standarddevation(coun)
            self.statistical[coun][6] = self.calculate_normal2nddiff(coun)
        return self.statistical[coun][6]

    def calculate_normal2nddiff(self,coun):
        """
        Calculates and stores the normalized second difference of the data for a specified channel.

        Args:
            coun (int): The channel number for which the normalized second difference is obtained.

        Returns:
            list: A list containing the normalized second difference values of the specified channel.

        Example:
            To get the normalized second difference of channel 0:
            ```python
            normal_2nd_diff = instance_of_reading.calculate_normal2nddiff(0)
            ```
        """
        # Method implementation
        ndiff2 = self.statistical[coun][5]/self.statistical[coun][2]
        return ndiff2
    
    def gethjorth(self, coun):
        """
        Retrieves or calculates and returns Hjorth parameters (Activity, Mobility, Complexity) for a specified channel.

        Args:
            coun (int): The channel number for which Hjorth parameters are obtained.

        Returns:
            list: A list containing Hjorth parameters [Activity, Mobility, Complexity] of the specified channel.

        Example:
            To get Hjorth parameters for channel 0:
            ```python
            hjorth_params = instance_of_reading.gethjorth(0)
            ```

        Note:
            Hjorth parameters describe the signal's activity, mobility, and complexity.
            - Activity: Represents the energy of the signal.
            - Mobility: Represents the mean frequency of the signal.
            - Complexity: Represents the bandwidth of the signal.
        """
        # Method implementation
        if (self.hjorth[coun][0] == None):
            self.hjorth[coun][0] = self.calculate_activity(coun)
        if (self.hjorth[coun][1] == None):
            self.hjorth[coun][1] = self.calculate_mobility(coun)
        if (self.hjorth[coun][2] == None):
            self.hjorth[coun][2] = self.calculate_complexity(coun)
        return self.hjorth[coun]
    
    def getmobility(self, coun):
        """
        Retrieves or calculates and returns the mobility of a specified channel.

        Args:
            coun (int): The channel number for which mobility is obtained.

        Returns:
            float: The mobility value of the specified channel.

        Example:
            To get the mobility of channel 0:
            ```python
            mobility_value = instance_of_reading.getmobility(0)
            ```

        Note:
            Mobility is a Hjorth parameter that represents the mean frequency of the signal.
            It is calculated using the standard deviation and mean of the signal.
        """
        if (self.statistical[coun][2] == None):
            if (self.statistical[coun][1] == None):
                self.statistical[coun][1] = self.calculate_mean(coun)
        if (self.hjorth[coun][1] == None):
            self.hjorth[coun][1] = self.calculate_mobility(coun)
        return self.hjorth[coun][1]
    
    def calculate_mobility(self, coun):
        """
        Calculates and stores the mobility of a specified channel.

        Args:
            coun (int): The channel number for which mobility is obtained.

        Returns:
            float: The mobility value of the specified channel.

        Example:
            To get the mobility of channel 0:
            ```python
            mobility_value = instance_of_reading.calculate_mobility(0)
            ```

        Note:
            Mobility is a Hjorth parameter that represents the mean frequency of the signal.
            It is calculated using the standard deviation and mean of the signal.
        """
        # Method implementation
        if self.derdata[coun] == None:
            self.derdata[coun] = self.getderdata(coun)
        nomsum = 0
        for index in self.data[coun]:
            nomsum = nomsum + pow((float(index)-self.statistical[coun][1]),2)
        sum = 0
        for index in self.derdata[coun]:
            sum = sum + float(index)
        mean = sum/len(self.derdata[coun])
        dersum = 0
        self.dermean[coun] = mean
        for index in self.derdata[coun]:
            dersum = dersum + pow((float(index)-mean),2)
        mob = math.sqrt(dersum/nomsum)
        return mob
    
    def getacomplexity(self,coun):
        """
        Retrieves or calculates and returns the complexity of a specified channel.

        Args:
            coun (int): The channel number for which complexity is obtained.

        Returns:
            float: The complexity value of the specified channel.

        Example:
            To get the complexity of channel 0:
            ```python
            complexity_value = instance_of_reading.getacomplexity(0)
            ```

        Note:
            Complexity is a Hjorth parameter that represents the signal's changeability.
            It is calculated using the standard deviation of the signal's first derivative.
        """
        # Method implementation
        if (self.statistical[coun][2] == None):
            if (self.statistical[coun][1] == None):
                self.statistical[coun][1] = self.calculate_mean(coun)
        if (self.hjorth[coun][1] == None):
            self.hjorth[coun][1] = self.calculate_mobility(coun)
        if (self.hjorth[coun][2] == None):
            self.hjorth[coun][2] = self.calculate_complexity(coun)
        return self.hjorth[coun][2]
    
    def calculate_complexity(self,coun):
        """
        Calculates and stores the complexity of a specified channel.

        Args:
            coun (int): The channel number for which complexity is obtained.

        Returns:
            float: The complexity value of the specified channel.

        Example:
            To get the complexity of channel 0:
            ```python
            complexity_value = instance_of_reading.calculate_complexity(0)
            ```

        Note:
            Complexity is a Hjorth parameter that represents the signal's changeability.
            It is calculated using the standard deviation of the signal's first derivative.
        """
        # Method implementation
        if self.derderdata[coun] == None:
            self.derderdata[coun] = self.getderderdata(coun)
        if self.derdata[coun] == None:
            self.derdata[coun] = self.getderdata(coun)
        nomsum = 0
        for index in self.derdata[coun]:
            nomsum = nomsum + pow((float(index)-self.dermean[coun]),2)
        sum = 0
        for index in self.derderdata[coun]:
            sum = sum + float(index)
        mean = sum/len(self.derderdata[coun])
        dersum = 0
        for index in self.derderdata[coun]:
            dersum = dersum + pow((float(index)-mean),2)
        mob = math.sqrt(dersum/nomsum)
        com = mob/self.hjorth[coun][1]
        return com

    def getactivity(self,coun):
        """
        Retrieves or calculates and returns the activity of a specified channel.

        Args:
            coun (int): The channel number for which activity is obtained.

        Returns:
            float: The activity value of the specified channel.

        Example:
            To get the activity of channel 0:
            ```python
            activity_value = instance_of_reading.getactivity(0)
            ```

        Note:
            Activity is a Hjorth parameter that represents the signal's energy or the overall
            magnitude of the signal. It is calculated using the signal's standard deviation.
        """
        # Method implementation
        if (self.hjorth[coun][0] == None):
            self.hjorth[coun][0] = self.calculate_activity(coun)
        return self.hjorth[coun][0]
    
    def calculate_activity(self,coun):
        """
        Calculates and stores the activity of a specified channel.

        Args:
            coun (int): The channel number for which activity is obtained.

        Returns:
            float: The activity value of the specified channel.

        Example:
            To get the activity of channel 0:
            ```python
            activity_value = instance_of_reading.calculate_activity(0)
            ```

        Note:
            Activity is a Hjorth parameter that represents the signal's energy or the overall
            magnitude of the signal. It is calculated using the signal's standard deviation.
        """
        # Method implementation
        if (self.statistical[coun][1] == None):
            self.statistical[coun][1] = self.calculate_mean(coun)
        sum = 0
        for index in self.data[coun]:
            sum = sum + pow((float(index)-self.statistical[coun][1]),2)
        act = sum/len(self.data[coun])
        return act
    
    def getnsi(self, num_segments, coun):
        """
        Retrieves or calculates and returns the Non-Stationary Index (NSI) of a specified channel.

        Args:
            num_segments (int): The number of segments used for calculating NSI.
            coun (int): The channel number for which NSI is obtained.

        Returns:
            float: The Non-Stationary Index value of the specified channel.

        Example:
            To get the NSI of channel 0 with 5 segments:
            ```python
            nsi_value = instance_of_reading.getnsi(5, 0)
            ```

        Note:
            The Non-Stationary Index (NSI) is a measure of how much a signal's statistical
            properties change over time. It is calculated by dividing the signal into segments
            and computing the ratio of the standard deviation of each segment to the overall
            standard deviation of the entire signal.
        """
        # Method implementation
        self.nsi[coun] = self.calculate_nsi(num_segments, coun)
        return self.nsi[coun]
    
    def calculate_nsi(self, num_segments, coun):
        """
        Calculates and stores the Non-Stationary Index (NSI) of a specified channel.

        Args:
            num_segments (int): The number of segments used for calculating NSI.
            coun (int): The channel number for which NSI is obtained.

        Returns:
            float: The Non-Stationary Index value of the specified channel.

        Example:
            To get the NSI of channel 0 with 5 segments:
            ```python
            nsi_value = instance_of_reading.calculate_nsi(5, 0)
            ```

        Note:
            The Non-Stationary Index (NSI) is a measure of how much a signal's statistical
            properties change over time. It is calculated by dividing the signal into segments
            and computing the ratio of the standard deviation of each segment to the overall
            standard deviation of the entire signal.
        """
        # Method implementation
        self.normdata[coun] = self.normarray(coun)
        array_size = len(self.data[coun])/num_segments

        segments = self.split(self.normdata[coun],array_size)
        proms = []

        for segment in segments:
            sumv = 0
            for number in segment:
                sumv = sumv + float(number)
            prom = sumv/len(segment)
            proms.append(prom)

        while (len(proms)>num_segments):
            proms.pop()
            
        sumv = 0
        for index in proms:
            sumv = sumv + float(index)
        mean = sumv/len(proms)
        
        sumv = 0
        for index in proms:
            sumv = sumv + pow((float(index)-mean),2)
        sd = math.sqrt(sumv/len(proms))

        return sd
    
    def normarray(self, coun):
        """
        Normalizes the values of the specified channel's data array.

        Args:
            coun (int): The channel number for which data normalization is performed.

        Returns:
            numpy.ndarray: The normalized data array for the specified channel.

        Example:
            To normalize the data in channel 0:
            ```python
            normalized_data = instance_of_reading.normarray(0)
            ```

        Note:
            This method normalizes the values in the data array of the specified channel.
            Normalization is achieved by subtracting the minimum value from each element
            and dividing the result by the range (maximum value - minimum value) of the data.

        Warning:
            This method assumes that the data in the specified channel is numeric.
        """
        # Method implementation
        data = np.array(self.data[coun]).astype(float)
        minv = min(data)
        data = data - minv
        maxv = max(data)
        if maxv != 0:
            return data/maxv
        else:
            return data
    
    def split(self, arr, size):
        """
        Splits an array into segments of the specified size.

        Args:
            arr (list or numpy.ndarray): The array to be split into segments.
            size (int): The size of each segment.

        Returns:
            list: A list of segments, each of size 'size'.

        Example:
            To split an array 'my_array' into segments of size 5:
            ```python
            segments = instance_of_reading.split(my_array, 5)
            ```

        Note:
            This method is useful for dividing an array into smaller, more manageable segments.

        Warning:
            This method assumes that the input 'arr' is an iterable (list or numpy.ndarray).

        """
        # Method implementation
        arrs = []
        while len(arr) > size:
            pice = arr[:int(size)]
            arrs.append(pice)
            arr   = arr[int(size):]
        arrs.append(arr)
        return arrs
    
    def gethoc(self, coun):
        """
        Calculates and retrieves the High Order Crossing (HOC) feature for a specific channel.

        Args:
            coun (int): The index of the channel for which to calculate the HOC feature.

        Returns:
            list: A list containing the High Order Crossing feature values for the specified channel.

        Example:
            To get the HOC feature for channel 0:
            ```python
            hoc_feature_channel_0 = instance_of_reading.gethoc(0)
            ```

        Note:
            High Order Crossing is a feature that represents the number of times the signal crosses its zero.

        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.

        """
        # Method implementation
        if (self.hoc[coun] == None):
            self.hoc[coun] = self.calculate_hoc(coun)
        return self.hoc[coun]

    def calculate_hoc(self, coun):
        """
        Calculates and stores the High Order Crossing (HOC) feature for a specific channel.

        Args:
            coun (int): The index of the channel for which to calculate the HOC feature.

        Returns:
            list: A list containing the High Order Crossing feature values for the specified channel.

        Example:
            To get the HOC feature for channel 0:
            ```python
            hoc_feature_channel_0 = instance_of_reading.calculate_hoc(0)
            ```

        Note:
            High Order Crossing is a feature that represents the number of times the signal crosses its zero.

        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.

        """
        # Method implementation
        hoc = []
        temphoc = self.data[coun]
        for index in range(0,10):
            revtemp = temphoc
            temphoc = []
            for ind in range (0,len(revtemp)-1):
                temphoc.append(float(revtemp[ind+1])-float(revtemp[ind]))
            masked = []
            for indx in temphoc:
                if (indx>=0):
                    masked.append(1.0)
                else:
                    masked.append(0.0)
            cross = 0
            for i in range(0,(9 - index)):
                masked.pop(0)
            for inde in range(0,len(masked)-1):
                cross = cross + pow(masked[inde+1]-masked[inde],2)
            hoc.append(cross)
        return hoc

############################################################################################

    def getallfreq(self, coun):
        """
        Calculates and retrieves all frequency domain features for a specific channel.

        Args:
            coun (int): The index of the channel for which to calculate the frequency features.

        Returns:
            list: A list containing all frequency domain feature values for the specified channel.

        Example:
            To get all frequency features for channel 0:
            ```python
            all_freq_features_channel_0 = instance_of_reading.getallfreq(0)
            ```

        Note:
            This method includes features such as mean, minimum, maximum, and standard deviation for different frequency bands.

        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.

        """
        # Method implementation
        if (self.allfreq[coun] == None):
             a = self.getbp_delta(coun)
             b = self.getbp_tetha(coun)
             c = self.getbp_salfa(coun)
             d = self.getbp_alfa(coun)
             e = self.getbp_beta(coun)
             f = self.getbp_gamma(coun)
             g = self.get_ratio_betaalfa(coun)
             h = self.getbp_diffentro(coun)
             all = []
             all.extend(a)
             all.extend(b)
             all.extend(c)
             all.extend(d)
             all.extend(e)
             all.extend(f)
             all.append(g)
             all.extend(h)
             self.allfreq[coun] = all
             self.labelsfreq = ['delta mean','delta min','delta max','delta std','tetha mean','tetha min','tetha max','tetha std','slow alfa mean','slow alfa min','slow alfa max','slow alfa std','alfa mean','alfa min','alfa max','alfa std','beta mean','beta min','beta max','beta std','gamma mean','gamma min','gamma max','gamma std','ratio alfa beta','Diff entropy delta','Diff entropy tetha','Diff entropy slow alfa','Diff entropy alfa','Diff entropy beta','Diff entropy gamma']
        return self.allfreq[coun]
   
    def getallbp(self, coun):
        """
        Calculates and retrieves all bandpower features for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the bandpower features.
    
        Returns:
            list: A list containing all bandpower feature values for the specified channel.
    
        Example:
            To get all bandpower features for channel 0:
            ```python
            all_bandpower_channel_0 = instance_of_reading.getallbp(0)
            ```
    
        Note:
            This method includes bandpower features such as mean, minimum, maximum, and standard deviation for different frequency bands.
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.allbp[coun] == None):
             a = self.getbp_delta(coun)
             b = self.getbp_tetha(coun)
             c = self.getbp_salfa(coun)
             d = self.getbp_alfa(coun)
             e = self.getbp_beta(coun)
             f = self.getbp_gamma(coun)
             g = self.getbp_diffentro(coun)
             all = []
             all.extend(a)
             all.extend(b)
             all.extend(c)
             all.extend(d)
             all.extend(e)
             all.extend(f)
             all.extend(g)
             self.allbp[coun] = all
             self.labelsbp = ['delta mean','delta min','delta max','delta std','tetha mean','tetha min','tetha max','tetha std','slow alfa mean','slow alfa min','slow alfa max','slow alfa std','alfa mean','alfa min','alfa max','alfa std','beta mean','beta min','beta max','beta std','gamma mean','gamma min','gamma max','gamma std']
        return self.allbp[coun]

    def getfourier(self, coun):
        """
        Calculates the Fourier transform of the data for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Fourier transform.
    
        Returns:
            numpy.ndarray: An array containing the absolute values of the Fourier transform of the data.
    
        Example:
            To get the Fourier transform for channel 0:
            ```python
            fourier_transform_channel_0 = instance_of_reading.getfourier(0)
            ```
    
        Note:
            This method uses the Fast Fourier Transform (FFT) to calculate the frequency components of the input data.
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        npdata = np.asarray(self.data[coun])
        fourier = fft(npdata)
        return np.absolute(fourier)
    
    # método complementario para calcular el rango de frecuencia por bin
    def getfreqs(self, coun, fs=200):
        """
        Calculates the frequency range per bin for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the frequency range.
            fs (float, optional): The sampling frequency. Default is 200 Hz.
    
        Returns:
            list: A list containing the frequency values corresponding to each bin.
    
        Example:
            To get the frequency range for channel 0 with a sampling frequency of 250 Hz:
            ```python
            frequency_range_channel_0 = instance_of_reading.getfreqs(0, fs=250)
            ```
    
        Note:
            This method assumes that the data is sampled at a constant rate. Default is 200 Hz.
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        freqs = []
        #fs = 200.0
        for bin in range(0,len(self.data[coun])):
            freqs.append((bin+1)*fs/len(self.data[coun]))
        return freqs
    
    def getbp_delta(self, coun):
        """
        Calculates the Band Power in the delta frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing the mean, minimum, maximum, and standard deviation of the Band Power in the delta range.
    
        Example:
            To get the Band Power in the delta range for channel 0:
            ```python
            delta_band_power_channel_0 = instance_of_reading.getbp_delta(0)
            ```
     
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
        
        """
        # Method implementation
        if (self.bp_delta[coun][0] == None ):
            self.bp_delta[coun] = self.calculatebp(1,4, coun)
        return self.bp_delta[coun]

    def getbp_tetha(self, coun):
        """
        Calculates the Band Power in the theta frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing the mean, minimum, maximum, and standard deviation of the Band Power in the theta range.
    
        Example:
            To get the Band Power in the theta range for channel 0:
            ```python
            theta_band_power_channel_0 = instance_of_reading.getbp_tetha(0)
            ```
     
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.bp_tetha[coun][0] == None ):
            self.bp_tetha[coun] = self.calculatebp(4,8, coun)
        return self.bp_tetha[coun]

    def getbp_salfa(self, coun):
        """
        Calculates the Band Power in the slow alpha frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing the mean, minimum, maximum, and standard deviation of the Band Power in the slow alpha range.
    
        Example:
            To get the Band Power in the slow alpha range for channel 0:
            ```python
            salfa_band_power_channel_0 = instance_of_reading.getbp_salfa(0)
            ```
     
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.bp_salfa[coun][0] == None ):
            self.bp_salfa[coun] = self.calculatebp(8,10, coun)
        return self.bp_salfa[coun]

    def getbp_alfa(self, coun):
        """
        Calculates the Band Power in the alpha frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing the mean, minimum, maximum, and standard deviation of the Band Power in the alpha range.
    
        Example:
            To get the Band Power in the alpha range for channel 0:
            ```python
            alfa_band_power_channel_0 = instance_of_reading.getbp_alfa(0)
            ```
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.bp_alfa[coun][0] == None ):
            self.bp_alfa[coun] = self.calculatebp(8,12, coun)
        return self.bp_alfa[coun]

    def getbp_beta(self, coun):
        """
        Calculates the Band Power in the beta frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing the mean, minimum, maximum, and standard deviation of the Band Power in the beta range.
    
        Example:
            To get the Band Power in the beta range for channel 0:
            ```python
            beta_band_power_channel_0 = instance_of_reading.getbp_beta(0)
            ```
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.bp_beta[coun][0] == None ):
            self.bp_beta[coun] = self.calculatebp(12,30, coun)
        return self.bp_beta[coun]

    def getbp_gamma(self, coun):
        """
        Calculates the Band Power in the gamma frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing the mean, minimum, maximum, and standard deviation of the Band Power in the gamma range.
    
        Example:
            To get the Band Power in the gamma range for channel 0:
            ```python
            gamma_band_power_channel_0 = instance_of_reading.getbp_gamma(0)
            ```
            
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.bp_gamma[coun][0] == None ):
            self.bp_gamma[coun] = self.calculatebp(30,64, coun)
        return self.bp_gamma[coun]
    
    def get_ratio_betaalfa(self, coun):
        """
        Calculates the ratio of Band Power in the beta frequency range to the Band Power in the alfa frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the ratio.
    
        Returns:
            float: The ratio of Band Power in the beta range to the Band Power in the alfa range.
    
        Example:
            To get the ratio of Band Power in the beta range to the Band Power in the alfa range for channel 0:
            ```python
            ratio_betaalfa_channel_0 = instance_of_reading.get_ratio_betaalfa(0)
            ```
            
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.bp_beta[coun][0] == None ):
            self.bp_beta[coun] = self.calculatebp(12,30, coun)
        if (self.bp_alfa[coun][0] == None ):
            self.bp_alfa[coun] = self.calculatebp(8,12, coun)
        if (self.ratio_ba[coun] == None):
            self.ratio_ba[coun] = self.calculate_ratio(coun)
        return self.ratio_ba[coun]
    
    def calculate_ratio(self, coun):
        """
        Calculates the ratio of Band Power in the beta frequency range to the Band Power in the alfa frequency range for a specific channel.
    
        Args:
            coun (int): The index of the channel for which to calculate the ratio.
    
        Returns:
            float: The ratio of Band Power in the beta range to the Band Power in the alfa range.
    
        Example:
            To calculate the ratio of Band Power in the beta range to the Band Power in the alfa range for channel 0:
            ```python
            ratio_channel_0 = instance_of_reading.calculate_ratio(0)
            ```
    
        Note:
            This method assumes that the Band Power for beta and alfa frequencies has been previously computed using the `calculatebp` method.
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        return self.bp_beta[coun][0]/self.bp_alfa[coun][0]
    
    def calculatebp(self, minp, maxp, coun):
        """
        Calculates the Band Power in a specific frequency range for a given channel.
    
        Args:
            minp (int): The minimum frequency of the desired range.
            maxp (int): The maximum frequency of the desired range.
            coun (int): The index of the channel for which to calculate the Band Power.
    
        Returns:
            list: A list containing statistical measures of the Band Power, including mean, minimum, maximum, and standard deviation.
    
        Example:
            To calculate the Band Power in the delta frequency range for channel 0:
            ```python
            delta_band_power_channel_0 = instance_of_reading.calculatebp(1, 4, 0)
            ```
    
        Note:
            This method assumes that the frequency range specified by 'minp' and 'maxp' falls within the available frequency range.
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if type(self.fdata[coun]) != list:
            self.fdata[coun] = self.getfourier(coun)
        if self.freqs[coun] == None:
            self.freqs[coun] = self.getfreqs(coun)
        wave = [None]*4
        tempwave = []
        for i in range(0,len(self.data[coun])):
            if (self.freqs[coun][i]>minp and self.freqs[coun][i]<maxp):
                tempwave.append(self.fdata[coun][i])
        if (maxp==4):
            self.delta[coun] = tempwave
        if (maxp==8):
            self.tetha[coun] = tempwave
        if (maxp==10):
            self.salfa[coun] = tempwave
        if (maxp==12):
            self.alfa[coun] = tempwave
        if (maxp==30):
            self.beta[coun] = tempwave
        if (maxp==64):
            self.gamma[coun] = tempwave
        wave[0] = np.mean(tempwave)
        wave[1] = min(tempwave)
        wave[2] = max(tempwave)
        wave[3] = np.std(tempwave)
        return wave
    
    def getbp_diffentro(self, coun):
        """
        Returns the Differential Entropy of Band Powers for a given channel.
    
        Args:
            coun (int): The index of the channel for which to calculate Differential Entropy.
    
        Returns:
            list: A list containing the Differential Entropy values for different frequency bands.
    
        Example:
            To calculate the Differential Entropy for Band Powers of channel 0:
            ```python
            differential_entropy_channel_0 = instance_of_reading.getbp_diffentro(0)
            ```
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        if (self.diffentro[coun][0] == None ):
            self.getbp_delta(coun)
            self.getbp_tetha(coun)
            self.getbp_salfa(coun)
            self.getbp_alfa(coun)
            self.getbp_beta(coun)
            self.getbp_gamma(coun)
            self.diffentro[coun] = self.calculate_Differential_Entropy(coun)
        return self.diffentro[coun]

    def calculate_Differential_Entropy(self, coun):
        """
        Calculates the Differential Entropy of Band Powers for a given channel.
    
        Args:
            coun (int): The index of the channel for which to calculate Differential Entropy.
    
        Returns:
            list: A list containing the Differential Entropy values for different frequency bands.
    
        Example:
            To calculate the Differential Entropy for Band Powers of channel 0:
            ```python
            differential_entropy_channel_0 = instance_of_reading.calculate_Differential_Entropy(0)
            ```
    
        Warning:
            This method assumes that the input 'coun' is a valid index for the available channels.
    
        """
        # Method implementation
        arrtemp = []
        bands = [self.delta[coun],self.tetha[coun],self.salfa[coun],self.alfa[coun],self.beta[coun],self.gamma[coun]]
        for band in bands:
            arrtemp.append(self.calculate_Differential_Entropy_2(band))
        return arrtemp

    def calculate_Differential_Entropy_2(self, Band):
        """
        Calculates the Differential Entropy for a given frequency band.
    
        Args:
            Band (list): The list of values representing the frequency band.
    
        Returns:
            float: The calculated Differential Entropy for the given frequency band.
    
        Example:
            To calculate the Differential Entropy for a specific frequency band:
            ```python
            band_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            differential_entropy_band = instance_of_reading.calculate_Differential_Entropy_2(band_values)
            ```
    
        Note:
            This method uses the formula for Differential Entropy, assuming the input 'Band' represents
            the values of a frequency band.
    
        Warning:
            Ensure that the input 'Band' is a valid list of numerical values representing a frequency band.
    
        """
        # Method implementation
        for i in Band:
            Dif_Ent_Band = ((0.5) * np.log(i)) + ((0.5)*np.log(i)*((2*np.pi*np.e)/self.samples))
        return Dif_Ent_Band
    
############################################################################################

    # método que regresa una lista con todos los features de tiempo-frecuencia
    def getalltimefreq(self, level, coun):
        """
         Retrieves a list of all time-frequency features for a given signal.
     
         Args:
             level (int): The level of time-frequency features to calculate.
             coun (int): The channel index for which the features are calculated.
     
         Returns:
             list: A list containing all time-frequency features, including RMS and REE values.
     
         Example:
             To obtain all time-frequency features for a specific channel and level:
             ```python
             channel_index = 0
             time_freq_level = 3
             time_freq_features = instance_of_reading.getalltimefreq(time_freq_level, channel_index)
             ```
     
         Note:
             This method combines RMS and REE features for the specified channel and time-frequency level.
     
         Warning:
             Ensure that the 'level' and 'coun' parameters are valid for the signal and channel.
     
         """
         # Method implementation
        if (self.alltimefreq[coun] == None):
             z = self.getrms(level, coun)
             a = self.getree(coun)
             all = []
             all.append(z)
             all.extend(a)
             self.alltimefreq[coun] = all
             self.labelstimefreq = ['rms','ree delta','ree tetha','ree alfa','ree beta','ree gamma']
        return self.alltimefreq[coun]
   
    def getrms(self, level, coun):
        """
        Returns the Root Mean Square (RMS) value for a specific channel and time-frequency level.
    
        Args:
            level (int): The level of time-frequency features to consider for RMS calculation.
            coun (int): The channel index for which the RMS value is calculated.
    
        Returns:
            float: The Root Mean Square (RMS) value for the specified channel and time-frequency level.
    
        Example:
            To obtain the RMS value for a specific channel and time-frequency level:
            ```python
            channel_index = 0
            time_freq_level = 3
            rms_value = instance_of_reading.getrms(time_freq_level, channel_index)
            ```
    
        Note:
            The RMS value is calculated based on the specified time-frequency level and channel.
    
        Warning:
            Ensure that the 'level' and 'coun' parameters are valid for the signal and channel.
    
        """
        # Method implementation
        self.rms[coun] = self.calculate_rms(level, coun)
        return self.rms[coun]

    def calculate_rms(self, levels, coun):
        """
        Calculates the Root Mean Square (RMS) value for a specific channel and wavelet decomposition levels.
    
        Args:
            levels (int): The number of wavelet decomposition levels to consider for RMS calculation.
            coun (int): The channel index for which the RMS value is calculated.
    
        Returns:
            float: The Root Mean Square (RMS) value for the specified channel and wavelet decomposition levels.
    
        Example:
            To obtain the RMS value for a specific channel and wavelet decomposition levels:
            ```python
            channel_index = 0
            wavelet_levels = 4
            rms_value = instance_of_reading.calculate_rms(wavelet_levels, channel_index)
            ```
    
        Note:
            The RMS value is calculated based on the specified number of wavelet decomposition levels and channel.
    
        Warning:
            Ensure that the 'levels' and 'coun' parameters are valid for the signal and channel.
    
        """
        # Method implementation
        top = 0
        coeffs = pywt.wavedec(self.data[coun], 'db4', level=levels)
        coeffs.pop(0)
        for c in coeffs:
            for d in c:
                a = math.pow(d,2)
                top = top + a
        botton = 0
        for i in range(1,levels + 1):
            botton = botton + i
        rms = math.sqrt(top/botton)
        return rms
   
    def getree(self, coun):
        """
        Retrieves or calculates the Recursive Energy Efficiency (REE) for a specific channel.
    
        Args:
            coun (int): The channel index for which the REE is retrieved or calculated.
    
        Returns:
            list: A list containing REE values for different frequency bands (delta, theta, alpha, beta, gamma).
    
        Example:
            To obtain the REE values for a specific channel:
            ```python
            channel_index = 0
            ree_values = instance_of_reading.getree(channel_index)
            ```
    
        Note:
            If the REE values for the specified channel have not been calculated yet, the method will calculate and store them.
    
        Warning:
            Ensure that the 'coun' parameter is valid for the signal and channel.
    
        """
        # Method implementation
        if (self.ree[coun][0] == None):
            self.ree[coun] = self.calculate_ree(coun)
        return self.ree[coun]
    
    def calculate_ree(self,coun):
        """
        Calculates the Recursive Energy Efficiency (REE) for a specific channel.
    
        Args:
            coun (int): The channel index for which the REE is calculated.
    
        Returns:
            list: A list containing REE values for different frequency bands (delta, theta, alpha, beta, gamma).
    
        Example:
            To obtain the REE values for a specific channel:
            ```python
            channel_index = 0
            ree_values = instance_of_reading.calculate_ree(channel_index)
            ```
    
        Note:
            The REE values are calculated based on the signal of the specified channel.
    
        Warning:
            Ensure that the 'coun' parameter is valid for the signal and channel.
    
        """
        # Method implementation
        temp = [None]*5
        ebands = []
        coeffs = pywt.wavedec(self.data[coun], 'db4', level=6)
        for xd in range(0,5):
            sum = 0
            for i in coeffs[xd]:
                sum = sum + math.pow(i,2)
            sum = math.sqrt(sum)
            ebands.append(sum)                
        etotal = ebands[2] + ebands[3] + ebands[4]
        for i in range (0,5):
            temp[i] = ebands[i]/etotal
        return temp

############################################################################################
    
    def getallall(self,num,level,coun):
        """
        Retrieves or calculates a comprehensive list of features, combining time, frequency, and time-frequency domain features.
        
        Args:
            num (int): A parameter needed for feature calculation.
            level (int): The level parameter for time-frequency feature calculation.
            coun (int): The channel index for which features are retrieved or calculated.
        
        Returns:
            list: A list containing a comprehensive set of features for the specified channel.
        
        Example:
            To obtain a list of features for a specific channel:
            ```python
            channel_index = 0
            features_list = instance_of_reading.getallall(num_value, level_value, channel_index)
            ```
        
        Note:
            This method combines features obtained from the time domain, frequency domain, and time-frequency domain.
        
        Warning:
            Ensure that the 'coun' parameter is valid for the signal and channel.
        
        """
        # Method implementation
        self.getall(num, coun)
        self.getallfreq(coun)
        self.getalltimefreq(level,coun)
        all = []
        all.extend(self.all[coun])
        all.extend(self.allfreq[coun])
        all.extend(self.alltimefreq[coun])
        self.labelsall = []
        self.labelsall.extend(self.labelstime)
        self.labelsall.extend(self.labelsfreq)
        self.labelsall.extend(self.labelstimefreq)
        self.allfeatures[coun] = all
        return self.allfeatures[coun]

############################################################################################

    def getdiffasym(self, chan1, chan2, num, level, feature=None):
        """
        Calculates the differial asymmetry between feature sets of two different channels.
        
        Args:
            chan1 (int): The index of the first channel.
            chan2 (int): The index of the second channel.
            num (int): A parameter needed for feature calculation.
            level (int): The level parameter for time-frequency feature calculation.
            feature (int, optional): The index of a specific feature to calculate the difference. 
                If not provided, the method calculates the difference for all features.
        
        Returns:
            list or float: If 'feature' is not specified, a list containing the differences for all features is returned.
                           If 'feature' is specified, a float representing the difference for the specified feature is returned.
        
        Example:
            To calculate the difference for all features between two channels:
            ```python
            channel_index1 = 0
            channel_index2 = 1
            num_value = 10
            level_value = 3
            differences_list = instance_of_reading.getdiffasym(channel_index1, channel_index2, num_value, level_value)
            ```
        
            To calculate the difference for a specific feature between two channels:
            ```python
            feature_index = 3
            specific_difference = instance_of_reading.getdiffasym(channel_index1, channel_index2, num_value, level_value, feature_index)
            ```
        
        Note:
            This method is useful for comparing features between different channels.
        
        Warning:
            Ensure that the 'chan1' and 'chan2' parameters are valid channel indices for the signal.
        
        """
        # Method implementation
        f1 = self.getallall(num,level,chan1)
        f2 = self.getallall(num,level,chan2)
        sol = []
        if feature == None:
            for i in range(0,len(f1)):
                sol.append(f1[i]-f2[i])
        else:
            sol = f1[feature] - f2[feature]
        return sol
    
    def getratioasym(self, chan1, chan2, num, level, feature=None):
        """
        Calculates the ratio or rational asymmetry between feature sets of two different channels.
        
        Args:
            chan1 (int): The index of the first channel.
            chan2 (int): The index of the second channel.
            num (int): A parameter needed for feature calculation.
            level (int): The level parameter for time-frequency feature calculation.
            feature (int, optional): The index of a specific feature to calculate the ratio. 
                If not provided, the method calculates the ratio for all features.
        
        Returns:
            list or float: If 'feature' is not specified, a list containing the ratios for all features is returned.
                           If 'feature' is specified, a float representing the ratio for the specified feature is returned.
        
        Example:
            To calculate the ratio for all features between two channels:
            ```python
            channel_index1 = 0
            channel_index2 = 1
            num_value = 10
            level_value = 3
            ratios_list = instance_of_reading.getratioasym(channel_index1, channel_index2, num_value, level_value)
            ```
        
            To calculate the ratio for a specific feature between two channels:
            ```python
            feature_index = 3
            specific_ratio = instance_of_reading.getratioasym(channel_index1, channel_index2, num_value, level_value, feature_index)
            ```
        
        Note:
            This method is useful for comparing features between different channels, especially when dealing with ratios.
        
        Warning:
            Ensure that the 'chan1' and 'chan2' parameters are valid channel indices for the signal.
            Ensure that denominator values are not zero to avoid division by zero errors.
        
        """
        # Method implementation
        f1 = self.getallall(num,level,chan1)
        f2 = self.getallall(num,level,chan2)
        sol = []
        if feature == None:
            for i in range(0,len(f1)):
                if f2[i] != 0:
                    sol.append(f1[i]/f2[i])
                else:
                    sol.append(0)
        else:
            if f2[feature] != 0:
                sol = f1[feature] / f2[feature]
            else:
                sol = 0
        return sol
    
    # diff asymmetry
    def getdiffasymstats(self, chan1, chan2, feature=None):
        """
        Calculates the difference or differencial asymmetry between statistical feature sets of two different channels.
        
        Args:
            chan1 (int): The index of the first channel.
            chan2 (int): The index of the second channel.
            feature (int, optional): The index of a specific feature to calculate the difference. 
                If not provided, the method calculates the difference for all features.
        
        Returns:
            list or float: If 'feature' is not specified, a list containing the differences for all features is returned.
                           If 'feature' is specified, a float representing the difference for the specified feature is returned.
        
        Example:
            To calculate the difference for all statistical features between two channels:
            ```python
            channel_index1 = 0
            channel_index2 = 1
            differences_list = instance_of_reading.getdiffasymstats(channel_index1, channel_index2)
            ```
        
            To calculate the difference for a specific statistical feature between two channels:
            ```python
            feature_index = 3
            specific_difference = instance_of_reading.getdiffasymstats(channel_index1, channel_index2, feature_index)
            ```
        
        Note:
            This method is useful for comparing statistical features between different channels.
        
        Warning:
            Ensure that the 'chan1' and 'chan2' parameters are valid channel indices for the signal.
        
        """
        # Method implementation
        f1 = self.getstatistical(chan1)
        f2 = self.getstatistical(chan2)
        sol = []
        if feature == None:
            for i in range(0,len(f1)):
                sol.append(f1[i]-f2[i])
        else:
            sol = f1[feature] - f2[feature]
        return sol
    
    def getratioasymstats(self, chan1, chan2, feature=None):
        """
        Calculates the ratio or rational asymmetry between statistical feature sets of two different channels.
        
        Args:
            chan1 (int): The index of the first channel.
            chan2 (int): The index of the second channel.
            feature (int, optional): The index of a specific feature to calculate the ratio. 
                If not provided, the method calculates the ratio for all features.
        
        Returns:
            list or float: If 'feature' is not specified, a list containing the ratios for all features is returned.
                           If 'feature' is specified, a float representing the ratio for the specified feature is returned.
        
        Example:
            To calculate the ratio for all statistical features between two channels:
            ```python
            channel_index1 = 0
            channel_index2 = 1
            ratios_list = instance_of_reading.getratioasymstats(channel_index1, channel_index2)
            ```
        
            To calculate the ratio for a specific statistical feature between two channels:
            ```python
            feature_index = 3
            specific_ratio = instance_of_reading.getratioasymstats(channel_index1, channel_index2, feature_index)
            ```
        
        Note:
            This method is useful for comparing the ratios of statistical features between different channels.
        
        Warning:
            Ensure that the 'chan1' and 'chan2' parameters are valid channel indices for the signal.
            Avoid division by zero when 'feature' is specified by ensuring that the corresponding feature values are non-zero.
        
        """
        # Method implementation
        f1 = self.getstatistical(chan1)
        f2 = self.getstatistical(chan2)
        sol = []
        if feature == None:
            for i in range(0,len(f1)):
                if f2[i] != 0:
                    sol.append(f1[i]/f2[i])
                else:
                    sol.append(0)
        else:
            if f2[feature] != 0:
                sol = f1[feature] / f2[feature]
            else:
                sol = 0
        return sol
    
    def getdiffasymbp(self, chan1, chan2, feature=None):
        """
        Calculates the difference or differencial asymmetry between frequency feature sets of two different channels.
        
        Args:
            chan1 (int): The index of the first channel.
            chan2 (int): The index of the second channel.
            feature (int, optional): The index of a specific feature to calculate the difference.
                If not provided, the method calculates the difference for all band power features.
        
        Returns:
            list or float: If 'feature' is not specified, a list containing the differences for all band power features is returned.
                           If 'feature' is specified, a float representing the difference for the specified feature is returned.
        
        Example:
            To calculate the difference for all band power features between two channels:
            ```python
            channel_index1 = 0
            channel_index2 = 1
            differences_list = instance_of_reading.getdiffasymbp(channel_index1, channel_index2)
            ```
        
            To calculate the difference for a specific band power feature between two channels:
            ```python
            feature_index = 3
            specific_difference = instance_of_reading.getdiffasymbp(channel_index1, channel_index2, feature_index)
            ```
        
        Note:
            This method is useful for comparing the differences of band power features between different channels.
        
        Warning:
            Ensure that the 'chan1' and 'chan2' parameters are valid channel indices for the signal.
        
        """
        # Method implementation
        f1 = self.getallbp(chan1)
        f2 = self.getallbp(chan2)
        sol = []
        if feature == None:
            for i in range(0,len(f1)):
                sol.append(f1[i]-f2[i])
        else:
            sol = f1[feature] - f2[feature]
        return sol
    
    def getratioasymbp(self, chan1, chan2, feature=None):
        """
        Calculates the ratio or rational asymmetry between frenquency feature sets of two different channels.
        
        Args:
            chan1 (int): The index of the first channel.
            chan2 (int): The index of the second channel.
            feature (int, optional): The index of a specific feature to calculate the ratio.
                If not provided, the method calculates the ratio for all band power features.
        
        Returns:
            list or float: If 'feature' is not specified, a list containing the ratios for all band power features is returned.
                           If 'feature' is specified, a float representing the ratio for the specified feature is returned.
        
        Example:
            To calculate the ratio for all band power features between two channels:
            ```python
            channel_index1 = 0
            channel_index2 = 1
            ratios_list = instance_of_reading.getratioasymbp(channel_index1, channel_index2)
            ```
        
            To calculate the ratio for a specific band power feature between two channels:
            ```python
            feature_index = 3
            specific_ratio = instance_of_reading.getratioasymbp(channel_index1, channel_index2, feature_index)
            ```
        
        Note:
            This method is useful for comparing the ratios of band power features between different channels.
        
        Warning:
            Ensure that the 'chan1' and 'chan2' parameters are valid channel indices for the signal.
            The method handles cases where the denominator is zero by setting the ratio to zero.
        
        """
        # Method implementation
        f1 = self.getallbp(chan1)
        f2 = self.getallbp(chan2)
        sol = []
        if feature == None:
            for i in range(0,len(f1)):
                if f2[i] != 0:
                    sol.append(f1[i]/f2[i])
                else:
                    sol.append(0)
        else:
            if f2[feature] != 0:
                sol = f1[feature] / f2[feature]
            else:
                sol = 0
        return sol
