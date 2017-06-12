from yahoo_finance import Share
import matplotlib.pyplot as plt
import pandas as pd

import pandas_datareader as pdr
import datetime
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

from Data import HistoricDataHandler 



symbol_list=['AAPL','GOOG']

date='2017-01-01'

HDH = HistoricDataHandler('event',symbol_list)

#HDH.open_quandl_data(date)
HDH._open_quandl_data(date)
HDH.update_bars
print HDH.get_latest_bars('GOOG',N=10)



#print symbol_data



#quandl.ApiConfig.api_key = '4ZyHzmP1Hp73xkygPzzL'


#data = quandl.get_table("WIKI/PRICES", rows=5)

#data=quandl.get("WIKI/AAPL.4",start_date='2016-10-10',end_date='2016-10-10')
#print data['Close'][0]

