import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import datetime
import numpy as np
import requests
from requests.packages.urllib3.exceptions import InsecurePlatformWarning
requests.packages.urllib3.disable_warnings(InsecurePlatformWarning)
import multiprocessing
from functools import partial
import urllib2
from bs4 import BeautifulSoup
import re
from requests.exceptions import HTTPError

start = datetime.datetime(2017, 3, 15)
end = datetime.datetime(2017, 4, 19)

test=pdr.get_data_yahoo('UTX',start,end)
dates=test.index

cl = pd.read_csv('companylist.csv')
cl_sorted=cl.sort(['MarketCap'],ascending=0)

cl10=cl_sorted.iloc[0:50]


def addPE(clist):
    p = multiprocessing.Pool(processes=4)
    shareinfos=np.array(p.map(perShareInfo,clist['Symbol']))
    clist['PE']=shareinfos[:,2]
    return clist

def perShareInfo(tckr):
        web='http://stockreports.nasdaq.edgar-online.com/'+tckr+'.html'
        try:
            r = requests.get(web)
            r.raise_for_status()
        except HTTPError:
            print [tckr+': Could not download page']
            PE=[]
            dividend=[]
            EPS=[]
        else:
            page=urllib2.urlopen(web)
            soup=BeautifulSoup(page)
            t=soup.findAll('table')[9]
            u=t.findAll('tr')[1]
            s=u.findAll('font',face="Arial, Helvetica, Verdana")
            EPS=extractDigits(s[1].find(text=True))
            dividend=extractDigits(s[2].find(text=True))
            PE=extractDigits(s[3].find(text=True))
            
        return EPS,dividend,PE
    
def extractDigits(string):
    val=[]
    for t in string.split():
        try:
            val.append(float(t))
        except ValueError:
            pass
    return map(float,val)


class share:
    def __init__(self,tckr):
        self.name=tckr
        self.sector=cl[cl['Symbol'].str.contains(tckr)]['Industry'].tolist()
        self.marketcap=cl[cl['Symbol'].str.contains(tckr)]['MarketCap'].tolist()
        self.EPS, self.dividend,self.PE=perShareInfo(tckr)
          
    def high(self,start,end):
        return pdr.get_data_yahoo(self.name,start,end)['High']

    def plotShare(self,start,end):
        high=self.high(start,end)
        dates=high.index
        plt.plot(dates,high,linewidth=3)
        plt.show()

class portfolio():
    def __init__(self):
        self.shares=[]
        self.cash=[]
        self.stockvalue=[]
        self.total_stockvalue=[]
        self.x_shares=[]
        self.n_shares=[]
        
    
    def addShare(self,tckr,inv):
        share_t=share(tckr)
        self.shares.append([share(tckr),inv])

    def totalStockvalue(self,start,end):
        test_high=self.shares[0][0].high(start,end)
        n_dates=len(test_high.index)
        self.n_shares=len(self.shares)
        self.stockvalue=np.zeros((n_dates,self.n_shares))
        for i in range(0,self.n_shares):
            inv=self.shares[i][1]
            high=self.shares[i][0].high(start,end)
            x_shares=np.round(inv/high[0])
            self.x_shares.append(x_shares)
            self.stockvalue[:,i]=np.array(high*x_shares)
        self.total_stockvalue=np.cumsum(self.stockvalue,axis=1)[:,-1]

    def plotPortfolio(self,start,end):

        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
        for i in range(len(tableau20)):    
            r, g, b = tableau20[i]    
            tableau20[i] = (r / 255., g / 255., b / 255.)    

        high=self.shares[0][0].high(start,end)
        dates=high.index
        if (len(self.stockvalue)==0):
            self.totalStockvalue(start,end)

        if (len(self.shares)==1):
            cumulative_stockvalue=np.cumsum(self.stockvalue,axis=1)
        else:
            cumulative_stockvalue=np.cumsum(self.stockvalue,axis=1)

        fig = plt.figure(figsize=(20,10))
        ax = plt.subplot(111)
        patch=[]

        for i in np.arange(0,self.n_shares)[: :-1]:
            tckr=self.shares[i][0].name
            ax.plot(dates,cumulative_stockvalue[:,i],color=tableau20[2*i])
            ax.fill_between(dates, 0, cumulative_stockvalue[:,i],color=tableau20[2*i])
            patch.append(mpatches.Patch(color=tableau20[2*i], label=tckr))
        
        box = ax.get_position()
        ax.legend(handles=patch,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

            
#perShareInfo2('AAPL')
###ncl=addPE(cl10)

Jakob=portfolio()
Jakob.addShare('AAPL',2000)
Jakob.addShare('MSFT',2000)
Jakob.addShare('GOOG',2000)
Jakob.plotPortfolio(start,end)
#sv=Jakob.totalStockvalue(start,end)
#print Jakob.stockvalue

