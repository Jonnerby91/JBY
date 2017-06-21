from yahoo_finance import Share
import matplotlib.pyplot as plt
import pandas as pd

import pandas_datareader as pdr
import datetime
import numpy as np
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.arima_process as ap
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from arch import arch_model
from contextlib import contextmanager
import sys, os
import requests
from requests.packages.urllib3.exceptions import InsecurePlatformWarning
requests.packages.urllib3.disable_warnings(InsecurePlatformWarning)
import quandl 
quandl.ApiConfig.api_key = '4ZyHzmP1Hp73xkygPzzL'

from Event import MarketEvent
from Event import Event
from Data import HistoricDataHandler 
from Portfolio import NaivePortfolio
from Strategy import BuyAndHoldStrategy
from Execution import SimulatedExecutionHandler

from Queue import Queue



events=Queue(maxsize=0)
#events.put(MarketEvent())


symbol_list=['MSFT','AAPL','GOOG']

date='2017-06-10'
HDH = HistoricDataHandler(events,symbol_list)
HDH._open_quandl_data(date)
HDH.update_bars()
strategy=BuyAndHoldStrategy(HDH,events)
port=NaivePortfolio(HDH,events,date)
execute=SimulatedExecutionHandler(events)
event=events.get()
strategy.calculate_signals(event)
events.task_done()

#HDH.update_bars()

#for k in range(0,10):
    #HDH.update_bars()
    #print HDH.latest_symbol_data['MSFT']
while HDH.continue_backtest==True:
    try:
        print HDH._get_new_bar('MSFT').next()
    except StopIteration:
        HDH.continue_backtest=False
    print HDH.continue_backtest
    #events.task_done()


"""
for k in range(0,100):
    if events.empty():
        
        HDH.update_bars()
        if HDH.latest_symbol_data['GOOG'][1][1] == old_date:
            events.task_done()
            print 'Task done'
            break
    event=events.get()
    if event.type == 'MARKET':
         port.update_timeindex(event)
         port.create_equity_curve_dataframe()
        # print port.equity_curve
    if event.type == 'SIGNAL':
        print event.type, event.datetime, event.symbol, event.signal_type
        port.update_signal(event)
    if event.type == 'ORDER':
        print event.type, event.symbol, event.order_type, event.quantity, event.direction
        execute.execute_order(event)
    if event.type == 'FILL':
        print event.timeindex
        port.update_fill(event)
    events.task_done()
    print k
    old_date=HDH.latest_symbol_data['MSFT'][1][1]
"""

#events.task_done()


#bars=HDH.get_latest_bars('MSFT')

#NP=NaivePortfolio(bars=HDH,events,start_date='2014-10-10')


#a=HDH._get_new_bar('AAPL')
#print HDH.symbol_data

#print symbol_data

#for i in range(0,10):
 #   HDH.update_bars()

#from Queue import Queue

#def do_stuff(q):
 # while not q.empty():
  #  print q.get()
   # q.task_done()

#q = Queue(maxsize=0)

#for x in range(20):
#  q.put(x)

#do_stuff(q)


#quandl.ApiConfig.api_key = '4ZyHzmP1Hp73xkygPzzL'


#data = quandl.get_table("WIKI/PRICES", rows=5)

#data=quandl.get("WIKI/AAPL.4",start_date='2016-10-10',end_date='2016-10-10')
#print data['Close'][0]

