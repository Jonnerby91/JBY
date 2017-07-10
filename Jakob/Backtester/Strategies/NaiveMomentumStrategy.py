import datetime
import numpy as np
import pandas as pd
import Queue

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import sys, os
from statsmodels.tsa.arima_model import ARIMA


from Event import SignalEvent

from Strategy import Strategy

class NaiveMomentumStrategy(Strategy):
    """
    This is an extremely simple strategy that goes LONG all of the 
    symbols as soon as a bar is received. It will never exit a position.

    It is primarily used as a testing mechanism for the Strategy class
    as well as a benchmark upon which to compare other strategies.
    """

    def __init__(self, bars, events,debug=0):
        """
        Initialises the buy and hold strategy.

        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        # Once buy & hold signal is given, these are set to True
        self.bought, self.short = self._calculate_initial_bought()
        self.debug = debug

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to False.
        """
        bought = {}
        short = {}
        for s in self.symbol_list:
            bought[s] = False
            short[s] = False
        return bought, short

    def calculate_signals(self, event):
        """
        For "Buy and Hold" we generate a single signal per symbol
        and then no additional signals. This means we are 
        constantly long the market from the date of strategy
        initialisation.

        Parameters
        event - A MarketEvent object. 
        """
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars(s, N=200)
                if bars is not None and bars != [] and len(bars) > 10:
                    hist_data = pd.DataFrame(bars).as_matrix()[:,3].tolist()
                    pred = self.generate_prediction(hist_data,30,1e-4)
                    if self.debug == 1:
                            print 'Predicting p=', pred
                    if pred == -1:
                        if self.short[s] == False:
                            signal = SignalEvent(bars[0][0], bars[0][1], 'SHORT')
                            self.events.put(signal)
                            self.short[s] = True
                        if self.bought[s] == True:
                            signal = SignalEvent(bars[0][0], bars[0][1], 'EXIT')
                            self.events.put(signal)
                            self.bought[s] = False
                    if pred == 1:
                        if self.short[s] == True:
                            signal = SignalEvent(bars[0][0], bars[0][1], 'EXIT')
                            self.events.put(signal)
                            self.short[s] = False
                        if self.bought[s] == False:
                            signal = SignalEvent(bars[0][0], bars[0][1], 'LONG')
                            self.events.put(signal)
                            self.bought[s] = True

    def generate_prediction(self,hist_data,N,k):
        N_mean = np.mean(hist_data[-N:])
        if hist_data[-1] > (1+k) * N_mean:
            pred = -1
        elif hist_data[-1] < (1-k) * N_mean:
            pred = 1
        else: 
            pred = 0
        return pred
                    
