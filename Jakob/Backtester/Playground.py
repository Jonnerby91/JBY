import quandl 
import pandas as pd
import numpy as np
from Features import SimpleFeatures
quandl.ApiConfig.api_key = '4ZyHzmP1Hp73xkygPzzL' 
date = '2014-01-01'

hist_data = quandl.get("WIKI/MSFT",start_date=date,end_date='2017-06-16')['Close'].as_matrix().tolist()

SF = SimpleFeatures('MSFT','bars')
dd = SF.Drawdown(hist_data)
print dd


