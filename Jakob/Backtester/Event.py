import numpy as np

class Event(object):
    # Provides an interface for all types of Events in the trader
    pass

class MarketEvent(Event):
    # Obtains the latest market data
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    # Generates a signal based on a Strategy 
    # Strength missing?
    def __init__(self,symbol, datetime, signal_type):
        self.type = 'SIGNAL'
        self.symbol=symbol
        self.datetime=datetime
        self.signal_type= signal_type
        self.strength = 1

class OrderEvent(Event):
    # Sends an Order for execution
    def __init__(self,symbol,order_type,quantity,direction):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        print "Order: Symbol = %s, Type = %s, Quantity = %s, Direction = %s" % \
            (self.symbol, self.order_type, self.quantity,self.direction)

class FillEvent(Event):
    # Returns stats on a filled order (buy or sell)
    def __init__(self, timeindex, symbol, exchange, quantity, direction, fill_cost, commission=None):
        """
        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from IB.
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = 0

        # Calculate commission
       # if commission is None:
        #    self.commission = self.calculate_ib_commission()
        #else:
         #   self.commission = commission

    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API, in USD.

        This does not include exchange or ECN fees.

        Based on "US API Directed Orders":
        https://www.interactivebrokers.com/en/index.php?f=commission&p=stocks2
        """
        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013 * self.quantity)
        else: # Greater than 500
            full_cost = max(1.3, 0.008 * self.quantity)
        full_cost = min(full_cost, 0.5 / 100.0 * self.quantity * self.fill_cost)
        return full_cost
