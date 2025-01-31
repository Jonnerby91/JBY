import datetime
import os, os.path
import pandas as pd
from abc import ABCMeta, abstractmethod
from Event import MarketEvent
from Event import Event
import quandl 
quandl.ApiConfig.api_key = '4ZyHzmP1Hp73xkygPzzL'

class DataHandler(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError("Not yet implemented")
        
    def update_bars(self):
        raise NotImplementedError("Not yet implemented")


class HistoricDataHandler(DataHandler):

    def __init__(self, events, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True     
        self.last_date={}

    def _open_quandl_data(self,date):
            """
            Opens the CSV files from the data directory, converting
            them into pandas DataFrames within a symbol dictionary.
            
            For this handler it will be assumed that the data is
            taken from DTN IQFeed. Thus its format will be respected.
            """
            comb_index = None
            for s in self.symbol_list:
                # Load the CSV file with no header information, indexed on date
                self.symbol_data[s] = quandl.get("WIKI/"+s,start_date=date,end_date='2017-06-16')

                # Combine the index to pad forward values
                if comb_index is None:
                    comb_index = self.symbol_data[s].index
                else:
                    comb_index.union(self.symbol_data[s].index)

                # Set the latest symbol_data to None
                self.latest_symbol_data[s] = []

            # Reindex the dataframes
            for s in self.symbol_list:
                self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows() 

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed as a tuple of 
        (sybmbol, datetime, open, low, high, close, volume).
        """
        for b in self.symbol_data[symbol]:
            yield tuple([symbol, b[0].to_pydatetime().strftime('%Y-%m-%d'), 
                        b[1][0], b[1][1], b[1][2], b[1][3], b[1][4]])

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print "That symbol is not available in the historical data set."
        else:
            return bars_list[-N:]            



    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            
            try:
                bar = self._get_new_bar(s).next()
            
            except StopIteration:
                self.continue_backtest = False
            
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())

class FXDataHandler(DataHandler):

    def __init__(self, events, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True     
        self.last_date={}

    def _open_data(self):
            """
            Opens the CSV files from the data directory, converting
            them into pandas DataFrames within a symbol dictionary.
            
            For this handler it will be assumed that the data is
            taken from DTN IQFeed. Thus its format will be respected.
            """
            comb_index = None
            for s in self.symbol_list:
                # Load the CSV file with no header information, indexed on date
                #self.symbol_data[s] = pd.read_csv('./Databases/FX/DAT_NT_'+s+'_T_ASK_201705.csv',delimiter=';',names=["Time","Ask"])
                ask_bid=pd.read_csv('./Databases/FX/Ask/DAT_NT_'+s+'_T_ASK_201705.csv',delimiter=';',usecols=[0,1])
                ask_bid[len(ask_bid.columns)]=pd.read_csv('./Databases/FX/Bid/DAT_NT_'+s+'_T_BID_201705.csv',delimiter=';',usecols=[1])
                self.symbol_data[s] = ask_bid
#quandl.get("WIKI/"+s,start_date=date,end_date='2017-06-16')

                # Combine the index to pad forward values
                if comb_index is None:
                    comb_index = self.symbol_data[s].index
                else:
                    comb_index.union(self.symbol_data[s].index)

                # Set the latest symbol_data to None
                self.latest_symbol_data[s] = []

            # Reindex the dataframes
            for s in self.symbol_list:
                self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows() 

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed as a tuple of 
        (sybmbol, datetime, open, low, high, close, volume).
        """
        for b in self.symbol_data[symbol]:
            year = int(b[1][0][0:4])
            month = int(b[1][0][4:6])
            day = int(b[1][0][6:8])
            hour = int(b[1][0][9:11])
            minute = int(b[1][0][11:13])
            second = int(b[1][0][13:15])
            dateandtime = datetime.datetime(year,month,day,hour,minute,second)
            yield tuple([symbol,dateandtime,b[1][1],b[1][2]])


#symbol, b[0].to_pydatetime().strftime('%Y-%m-%d'), 
 #                       b[1][0], b[1][1], b[1][2], b[1][3], b[1][4]])

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print "That symbol is not available in the historical data set."
        else:
            return bars_list[-N:]            



    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            
            try:
                bar = self._get_new_bar(s).next()
            
            except StopIteration:
                self.continue_backtest = False
            
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())
