# Declare the components with respective parameters
import sys
sys.path.append('/home/jonnerbyl/Documents/HF/Jakob/Backtester/Strategies')

import Queue
from Data import HistoricDataHandler, FXDataHandler
from NaiveMomentumStrategy import NaiveMomentumStrategy
from Portfolio import NaivePortfolio
from Execution import SimulatedExecutionHandler
from Event import MarketEvent
from Event import Event
import time
import matplotlib.pyplot as plt
import datetime


events=Queue.Queue(maxsize=0)
debug = 1

symbol_list=['EURUSD']
#start_date='2016-03-01'


bars = FXDataHandler(events,symbol_list)
bars._open_data()
bars.update_bars()

start_date = bars.get_latest_bars('EURUSD')[0][1]
end_date = datetime.datetime(2017,5,1,5,0,0)
current_date = start_date

strategy = NaiveMomentumStrategy(bars,events,debug)
#strategy = ARIMAGARCHStrategy(bars,events)
port = NaivePortfolio(bars,events,start_date,debug)
#broker = ExecutionHandler(..)
broker=SimulatedExecutionHandler(events)

# Initialise
#bars._open_quandl_data(start_date)


#for i in range(0,20000):
while current_date < end_date:
    # Update the bars (specific backtest code, as opposed to live trading)
    if bars.continue_backtest == True:
        bars.update_bars()
        current_date = bars.get_latest_bars('EURUSD')[0][1]
        #print bars.latest_symbol_data['MSFT'][-1][1]
    else:
        break
    
    # Handle the events
    while True:
        try:
            event = events.get(False)
            #events.task_done()
        except Queue.Empty:
            break
        else:
            if event is not None:
                if debug == 1 and event.type is not 'MARKET':
                        print event.type
                if event.type == 'MARKET':
                    strategy.calculate_signals(event)
                    port.update_timeindex(event)

                elif event.type == 'SIGNAL':
                    port.update_signal(event)
                    

                elif event.type == 'ORDER':
                    broker.execute_order(event)

                elif event.type == 'FILL':
                    port.update_fill(event)
            
    # 10-Minute heartbeat
    #time.sleep(1)


print port.output_summary_stats()

#port.create_equity_curve_dataframe()
returns = port.equity_curve['equity_curve']
plt.plot(returns.as_matrix()[1:-1])
plt.show()

#import pandas as pd
#from Strategy import ARIMAGARCHStrategy

#hist_data=pd.DataFrame(bars.get_latest_bars('MSFT',100)).as_matrix()[:,3].tolist()
#AGS=ARIMAGARCHStrategy(bars,events)
#AGS.generate_prediction(hist_data)
