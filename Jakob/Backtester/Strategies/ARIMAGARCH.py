from Strategy import Strategy

import datetime
import numpy as np
import pandas as pd
import Queue

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import sys, os
from statsmodels.tsa.arima_model import ARIMA


from Event import SignalEvent


class ARIMAGARCHStrategy(Strategy):
    
    def __init__(self,bars,events):
        """
        Initialise the ARIMA/GARCH Strategy
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        # Once buy & hold signal is given, these are set to True
        self.bought = self._calculate_initial_bought()
    
    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to False.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought

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
                if bars is not None and bars != [] and len(bars) > 199:
                    
                        hist_data = pd.DataFrame(bars).as_matrix()[:,3].tolist()
                        pred = self.generate_prediction(hist_data)
                        if self.bought[s] == False:
                            if pred == 1:
                                # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                                signal = SignalEvent(bars[0][0], bars[0][1], 'LONG')
                                self.events.put(signal)
                                self.bought[s] = True
                         #   if pred == -1:
                          ##      # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                                signal = SignalEvent(bars[0][0], bars[0][1], 'SHORT')
                            #    self.events.put(signal)
                             #   self.bought[s] = True
                        else:
                            if pred == -1:
                                signal = SignalEvent(bars[0][0], bars[0][1], 'EXIT')
                                self.events.put(signal)
                                self.bought[s] = False
                        

    def generate_prediction(self,hist_data):
        y=pd.Series(np.diff(np.log(np.array(hist_data)))).tolist()
        #print y
        AIC=1e5
        order=[0.0,0.0,0.0]
        for i in range(1,2):
            for j in range(1,2):
                for d in range(0,1):
                    #print i,d,j
                    try:
                        with suppress_stdout():
                            model2=ARIMA(y,order=(i,d,j))
                            model2_fit=model2.fit(disp=0)
                            caic=model2_fit.aic
                            if caic < AIC:
                                AIC=caic
                                order=[i,d,j]
                    except (ValueError,np.linalg.linalg.LinAlgError, ValueError) as e:
                        pass
         
        model=ARIMA(y,order=(order[0],order[1],order[2]))
        model_fit=model2.fit(disp=0)
        res=model2_fit.resid
        pred=model_fit.predict(start=-1, end=-1, dynamic=False) 
        print pred
        return np.sign(pred)
        
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


