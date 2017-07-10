from Data import FXDataHandler
import Queue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Strategy import BuyAndHoldStrategy, ARIMAGARCHStrategy


events=Queue.Queue(maxsize=0)


currency_list = ['EURUSD']

bars = FXDataHandler(events,currency_list)
strategy = BuyAndHoldStrategy(bars,events)


bars._open_data()
bars.update_bars()
event = events.get(False)
events.task_done()
strategy.calculate_signals(event)
event = events.get(False)
print event.type
print event.datetime
print event.signal_type



#N=1

#for i in range(0,N):
 #   FX.update_bars()


#FX.update_bars()
#FX.update_bars()

#b=FX.get_latest_bars('EURUSD',N)
#d=np.array(b)

#time = d[:,2]
#plt.plot(d[:,3])
#plt.show()

