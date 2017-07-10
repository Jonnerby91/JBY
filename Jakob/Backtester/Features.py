import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

class Features(object):
    """
    Provides a number of features as base for technical analysis of historical data.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def feature_list(self):
        """
        Provides a list of features
        """
        raise NotImplementedError("Should implement calculate_signals()")

class SimpleFeatures(Features):
    def __init__(self,symbol_list,bars):
        #self.hist_data = bars
        self.sl = symbol_list
        self.feature_list = self.feature_list()
        
    def Drawdown(self,hist_data):
        max_loc=np.argmax(hist_data)
        return len(hist_data)-max_loc
    
    def up_from_min(self,hist_data):
        min_loc=np.argmin(hist_data)
        return len(hist_data)-min_loc

    #def PeriodReturn(self,hist_data):
     #   return (hist_data[-1]-hist_data[0])/hist_data[0]
    
    
    def feature_list(self):
        return 0
    
