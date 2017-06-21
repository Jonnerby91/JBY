# Declare the components with respective parameters
#from Queue import Queue
import Queue
from Data import HistoricDataHandler
from Strategy import BuyAndHoldStrategy, ARIMAGARCHStrategy
from Portfolio import NaivePortfolio
from Execution import SimulatedExecutionHandler
from Event import MarketEvent
from Event import Event
import time
import matplotlib.pyplot as plt


events=Queue.Queue(maxsize=0)

symbol_list=['MSFT','AAPL','GOOG']
start_date='2016-03-01'

bars = HistoricDataHandler(events,symbol_list)
#strategy = BuyAndHoldStrategy(bars,events)
strategy = ARIMAGARCHStrategy(bars,events)
port = NaivePortfolio(bars,events,start_date)
#broker = ExecutionHandler(..)
broker=SimulatedExecutionHandler(events)

# Initialise
bars._open_quandl_data(start_date)

#for i in range(0,10):
while True:
    # Update the bars (specific backtest code, as opposed to live trading)
    if bars.continue_backtest == True:
        bars.update_bars()
        print bars.latest_symbol_data['MSFT'][-1][1]
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
#returns = port.equity_curve['equity_curve']
#plt.plot(returns.as_matrix()[1:-1])
#plt.show()

#import pandas as pd
#from Strategy import ARIMAGARCHStrategy

#hist_data=pd.DataFrame(bars.get_latest_bars('MSFT',100)).as_matrix()[:,3].tolist()
#AGS=ARIMAGARCHStrategy(bars,events)
#AGS.generate_prediction(hist_data)
