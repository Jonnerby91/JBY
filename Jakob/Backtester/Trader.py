# Declare the components with respective parameters
from Queue import Queue
from Data import HistoricDataHandler
from Strategy import BuyAndHoldStrategy
from Portfolio import NaivePortfolio


symbol_list=['MSFT','AAPL','GOOG']
date='2017-01-01'

bars = HistoricDataHandler(symbol_list=symbol_list)
strategy = BuyAndHoldStrategy(bars=bars,)
port = NaivePortfolio(bars=bars,start_date=date)
#broker = ExecutionHandler(..)

# Initialise
bars._open_quandl_data(date)

for i in range(0,10):
    # Update the bars (specific backtest code, as opposed to live trading)
    if bars.continue_backtest == True:
        bars.update_bars()
    else:
        break
    
    # Handle the events
    while True:
        try:
            event = events.get(False)
        except Queue.Empty:
            break
        else:
            if event is not None:
                if event.type == 'MARKET':
                    strategy.calculate_signals(event)
                    port.update_timeindex(event)

                elif event.type == 'SIGNAL':
                    port.update_signal(event)

               # elif event.type == 'ORDER':
                #    broker.execute_order(event)

                elif event.type == 'FILL':
                    port.update_fill(event)

    # 10-Minute heartbeat
    time.sleep(1)

