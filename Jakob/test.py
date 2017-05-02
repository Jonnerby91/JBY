import pandas_datareader as pdr
import matplotlib.pyplot as plt
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

def plotNormalized(clist):
    length=len(clist)
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    for i in range(0,length):
        name=clist['Symbol'].iloc[i]
        hist_data=pdr.get_data_yahoo(name,start,end)
        high=hist_data['High']/np.amax(hist_data['High'])
        ax.plot(dates, high, label=name,linewidth=3)
        box = ax.get_position()
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.plot(dates,high)
   
    plt.show()

def plotNormalizedSingle(tck):
    clist=cl[cl['Symbol'].str.contains(tck)]
    length=len(clist)
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    for i in range(0,length):
        name=clist['Symbol'].iloc[i]
        hist_data=pdr.get_data_yahoo(name,start,end)
        high=hist_data['High']/np.amax(hist_data['High'])
        ax.plot(dates, high, label=name,linewidth=3)
        box = ax.get_position()
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.plot(dates,high)
   
    plt.show()

def plotNormDiff(clist):
    length=len(clist)
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    for i in range(0,length):
        name=clist['Symbol'].iloc[i]
        hist_data=pdr.get_data_yahoo(name,start,end)
        high=hist_data['High']
        diffp=np.divide(np.diff(high),high[0:-1])
        ax.plot(dates[0:-1], diffp*100, label=name,linewidth=3)
        box = ax.get_position()
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    plt.show()

def RMSDiffCorr(clist):
    cols=len(clist)
    rows=len(dates)
    d=np.zeros((rows-1,cols))
    rms=np.zeros((cols,cols))
    
    for i in range(0,cols):
  
        name=clist['Symbol'].iloc[i]
        hist_data=pdr.get_data_yahoo(name,start,end)
        high=hist_data['High']
        diffp=np.divide(np.diff(high),high[0:-1])
        d[:,i]=diffp

    for j in range(0,cols):
        for k in range(j+1,cols):
            for t in range(0,rows-1):
                rms[j,k]+=(d[t,j]-d[t,k])**2
    

    return rms


def maxNowDiff(clist):
    cols=len(clist)
    rows=len(dates)
    maxdiff=np.zeros(cols)
    mindiff=np.zeros(cols)

   # p = multiprocessing.Pool(processes=4)
    #mad=p.map(worker,range(10))
        
    for i in range(0,cols):
        name=clist['Symbol'].iloc[i]
        hist_data=pdr.get_data_yahoo(name,start,end)
        high=hist_data['High']
        mnd=(high[-1]-np.amax(high))/np.amax(high)
        loc=np.argmax(high.tolist())
       # print 'l', loc
        mi=(np.amin(high[loc:-1])-np.amax(high))/np.amax(high)
        maxdiff[i]=mnd
        mindiff[i]=mi

    return maxdiff,mindiff

def maxNowDiffSingle(name):
   # name=clist['Symbol']
    hist_data=pdr.get_data_yahoo(name,start,end)
    high=hist_data['High']
    mnd=(high[-1]-np.amax(high))/np.amax(high)
    loc=np.argmax(high.tolist())
   # print 'l', loc
    mi=(np.amin(high[loc:-1])-np.amax(high))/np.amax(high)
    return mnd,mi

def recoveryBarPlots(clist):

    fig, ax = plt.subplots(figsize=((20,10)))
    tckrs=clist['Symbol'].tolist()
    p = multiprocessing.Pool(processes=4)
    #print cl10.iloc[:]
    mm=np.array(p.map(maxNowDiffSingle,cl10['Symbol']))
    mnd=mm[:,0]
    mi=mm[:,1]
    #mnd,mi=maxNowDiff(clist)
    x=np.linspace(0,len(clist),len(clist))
    width=0.35
    ax.bar(x,mi*100,width = width,color='r')
    ax.bar(x+width,mnd*100,width = width,color='b')
    ax.set_xticks(x + width )
    ax.set_xticklabels(tckrs)
    plt.show()

def addPE(clist):
    lc=len(clist)
    PE=np.array([])
    #for i in range(0,lc):
     #   name=clist['Symbol'].iloc[i]
      #  print i
       # PE=np.append(PE,share(name).PE)
    name=clist['Symbol']
    p = multiprocessing.Pool(processes=4)
    #print cl10.iloc[:]
    shareinfos=np.array(p.map(perShareInfo2,clist['Symbol']))
    PE=shareinfos[:,2]
    #print len(clist)
    #print len(PE)
    clist['PE']=PE
    print len(PE), len(clist) 
    return clist
        
def perShareInfo2(name):
        web='http://stockreports.nasdaq.edgar-online.com/'+name+'.html'
        print web
        #print name

        try:
            r = requests.get(web)
            r.raise_for_status()
        except HTTPError:
            print 'Could not download page'
            PE=[]
            dividend=[]
            EPS=[]
        else:
            page=urllib2.urlopen(web)
            soup=BeautifulSoup(page)
            t=soup.findAll('table')[9]
            u=t.findAll('tr')[1]
            s=u.findAll('font',face="Arial, Helvetica, Verdana")
            lens=len(s)
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
        self.EPS, self.dividend,self.PE=self.perShareInfo()
        
    def perShareInfo(self):
        web='http://stockreports.nasdaq.edgar-online.com/'+self.name+'.html'
        print self.name

        try:
            r = requests.get(web)
            r.raise_for_status()
        except HTTPError:
            print 'Could not download page'
            PE=np.nan
            dividend=np.nan
            EPS=np.nan
        else:
            page=urllib2.urlopen(web)
            soup=BeautifulSoup(page)
            t=soup.findAll('table')[9]
            u=t.findAll('tr')[1]
            s=u.findAll('font',face="Arial, Helvetica, Verdana")
            lens=len(s)
            EPS=self.extractDigits(s[1].find(text=True))
            dividend=self.extractDigits(s[2].find(text=True))
            PE=self.extractDigits(s[3].find(text=True))
            
        return EPS,dividend,PE
    
    def extractDigits(self,string):
        val=[]
        for t in string.split():
            try:
                val.append(float(t))
            except ValueError:
                pass
        return val
        
    def high(self,start,end):
        return pdr.get_data_yahoo(self.name,start,end)['High']

    def plotShare(self,start,end):
        high=self.high(start,end)
        dates=high.index
        plt.plot(dates,high,linewidth=3)
        plt.show()

class portfolio():
    def __init__(self,start,end):
        self.shares=[]
        self.n_shares=[]
        self.cash=[]
        self.stockvalue=np.array([])
        self.startend=[start,end]
    
    def addShare(self,tckr,inv):
        share_t=share(tckr)
        n_shares=np.round(inv/share_t.high(self.startend[0],self.startend[0])[0])
        self.n_shares.append(n_shares)
        if (len(self.shares)==0):
            self.shares.append(share_t)
            self.stockvalue=np.array(share_t.high(self.startend[0],self.startend[1])*n_shares)
        else:
            self.shares.append(share_t)
            self.stockvalue=np.c_[self.stockvalue,share_t.high(self.startend[0],self.startend[1])*n_shares]


    def plotPortfolio(self,start,end):
        high=self.shares[0].high(start,end)
        dates=high.index
        print self.stockvalue.shape
        cumulative_stockvalue=np.cumsum(self.stockvalue,axis=1)
        
        for i in range(0,len(self.shares)):
            plt.plot(dates,cumulative_stockvalue[:,i])
        plt.show()
        
    def totalStockvalue(self):
        return np.cumsum(self.stockvalue,axis=1)[:,-1]
        
        



#Microsoft=share('MSFT')
#Apple=share('AAPL')
#Google=share('GOOGL')
#Apple.plotShare(start,end)
#print Apple.marketcap



#print Jakob.shares[0].high(start,end)

#fig, ax = plt.subplots(figsize=((20,10)))


cl10=cl_sorted[0:50]
ct=addPE(cl10)
ct_sorted=ct.sort(['PE'],ascending=1)
print ct_sorted.head(10)

Jakob=portfolio(start,end)
#Jakob.addShare('AAPL',2000)
#Jakob.addShare('MSFT',2000)
#Jakob.addShare('GOOG',2000)
#print Jakob.stockvalue
#Jakob.plotPortfolio(start,end)
#print Jakob.totalStockvalue()[0]

#for k in range(0,10):
 #   tick=ct_sorted['Symbol'].iloc[k]

#Jakob.addShare(tick,1000)

#Jakob.plotPortfolio(start,end)
#print Jakob.totalStockvalue()
    

#    print s[i].find(text=True)



#print soup.body.findAll(text=re.compile('P/E'))

#result = re.search(r'P/E', str(soup), re.DOTALL)
#print(result.group(1))


#p = multiprocessing.Pool(processes=4)
#print cl10.iloc[:]
#x=np.array(p.map(maxNowDiffSingle,cl10['Symbol']))
#print x



#tckrs=cl10['Symbol'].tolist()

#prin#fig.canvas.draw()
#labels = [item.get_text() for item in ax.get_xticklabels()]
#labels[1] = 'Testing'
#labels=tckrs
#ax.set_xticklabels(labels)

#recoveryBarPlots(cl10)
#plotNormalizedSingle('TROW')
#plotNormDiff(cl_sorted.head(5))
#rms=RMSDiffCorr(cl_sorted.head(10))

#mnd,mi=maxNowDiff(cl10)
#x=np.linspace(0,len(cl10),len(cl10))
#width=0.35
#ax.bar(x,mi,width = width,color='r')
#ax.bar(x+width,mnd,width = width,color='b')
#ax.set_xticks(x + width )
#ax.set_xticklabels(tckrs)
#plt.plot(rms)
#plt.show()



#for i in range(0,5):
   # name=cl['Symbol'].iloc[i]
   # hist_data=pdr.get_data_yahoo(name,start,end)
  #  high=hist_data['High']/np.amax(hist_data['High'])
 #   plt.plot(dates,high)

#plt.show()
