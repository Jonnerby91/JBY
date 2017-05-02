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

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

start = datetime.datetime(2016, 1, 15)
end = datetime.datetime(2017, 4, 1)

def Pred(ticker,start,end):
    yclose=pdr.get_data_yahoo(ticker,start,end)['Close']
    #y=np.diff(np.log(np.array(yclose)))
    y=pd.Series(np.log(np.array(yclose))).pct_change()[1:].tolist()
    print y
    AIC=1e5
    order=[0.0,0.0,0.0]
    for i in range(1,5):
        for j in range(1,5):
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
    pred=model_fit.predict(start=0, end=0, dynamic=False)    

    with suppress_stdout():
        garch11=arch_model(res,p=1,q=1)
        garch11_fit=garch11.fit(disp=0)
        gres=garch11_fit.resid
        omega=garch11_fit.params['omega']
        alpha1=garch11_fit.params['alpha[1]']
        beta1=garch11_fit.params['beta[1]']
        cond_vol=garch11_fit.conditional_volatility[-1]
        forecast_vol=np.sqrt(omega+alpha1*gres[-1]**2+beta1*cond_vol**2)
   
    return pred*forecast_vol

ticklist=['GOOG','^FTSE','AAPL','FB']
flist=np.zeros(len(ticklist))
for i in range(0,len(ticklist)):
    f=Pred(ticklist[i],start,end)
    flist[i]=f
    print ticklist[i], f

start2 = datetime.datetime(2017, 3, 31)
end2 = datetime.datetime(2017, 4, 5)
ya=pdr.get_data_yahoo('GOOG',start2,end2)['Close']
