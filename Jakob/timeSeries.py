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

start1 = datetime.datetime(2016, 1, 15)
start2 = datetime.datetime(2017, 4, 1)
end1 = datetime.datetime(2017, 4, 1)
end2 = datetime.datetime(2017, 4, 21)

yclose=pdr.get_data_yahoo('^FTSE',start1,end1)['Close']
y=np.diff(np.log(np.array(yclose)))
t_close=pdr.get_data_yahoo('^FTSE',start2,end2)['Close']
test=np.diff(np.log(np.array(t_close)))

ar_coef = [0.2, -0.25]
ma_coef = [0.5, -0.3]
#y = ap.ArmaProcess(ar_coef, ma_coef,nobs=500).generate_sample(500)
#y=np.diff(np.diff(y))

#model=AR(data)
"""
AIC=1e5
order=[0.0,0.0,0.0]
for i in range(1,5):
    for j in range(1,5):
        for d in range(0,1):
            print i,d,j
            try:
                model2=ARIMA(y,order=(i,d,j))
                model2_fit=model2.fit(disp=0)
                caic=model2_fit.aic
                print caic
                if caic < AIC:
                    AIC=caic
                    
                    order[0]=i
                    order[1]=d
                    order[2]=j
            except (ValueError,np.linalg.linalg.LinAlgError) as e:
                pass

print AIC
print order

"""
order=[4,0,1]

model2=ARIMA(y,order=(order[0],order[1],order[2]))
model2_fit=model2.fit(disp=0)

res=model2_fit.resid
garch11=arch_model(res,p=1,q=1)
garch11_fit=garch11.fit(disp=0)
gres=garch11_fit.resid
omega=garch11_fit.params['omega']
alpha1=garch11_fit.params['alpha[1]']
beta1=garch11_fit.params['beta[1]']
cond_vol=garch11_fit.conditional_volatility[-1]

forecast_vol=np.sqrt(omega+alpha1*gres[-1]**2+beta1*cond_vol**2)
pred2=model2_fit.predict(start=0, end=0, dynamic=False)

print forecast_vol

#gredict=garch11_fit.forecast(start=len(y)-3,horizon=3, method='analytic')
#gr1=gredict.mean['h.01'].iloc[-1]


#pred2=model2_fit.predict(start=0, end=len(test), dynamic=False)
#sumpr=gr1+pred2

#plt.plot(sumpr)
#plt.plot(test)
#plt.show()

#plt.plot(acf(gres)**2)
#plt.plot(res**2)

#data,p=acorr_ljungbox(model2_fit.resid)
#pd.tools.plotting.autocorrelation_plot(model2_fit.resid)

#pred2=model2_fit.predict(start=0, end=len(test), dynamic=False)
#plt.plot(pred2,color='y')
#plt.plot(y)

#plt.show()

#params=model2_fit.params
#errors=model2.geterrors(params)
#conf=model2_fit.conf_int()

#print params
#print conf

#plt.plot(y)
#plt.show()

#print params
#print model2.geterrors(params)
#print output
#pred2=model2_fit.predict(start=len(y), end=len(y)+len(x)-1, dynamic=False)
#back=model2_fit.predict(start=0, end=len(x)-1, dynamic=False)

#plt.plot(x)
#plt.plot(back)
#pd.tools.plotting.autocorrelation_plot(y)
#plt.show()

#rand=np.random.uniform(1,10,1000)
#plt.plot(data)
#fig1=plt.figure()
#pd.tools.plotting.autocorrelation_plot(np.diff(data))

#model_fit=model.fit()
#predictions = model_fit.predict(start=len(data), end=len(data)+len(test)-1, dynamic=False)
#fig2=plt.figure()
#plt.plot(predictions*1.001)
#plt.plot(test)
#pd.tools.plotting.autocorrelation_plot(np.diff(rand))

#fig3=plt.figure()
#plt.plot(pd.DataFrame(model2_fit.resid))
#pd.tools.plotting.autocorrelation_plot(model2_fit.resid)
#plt.plot(pred2)
#plt.plot(test)

#plt.show()
