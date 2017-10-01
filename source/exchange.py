from stock import Stock, OUStock

class StockExchange(object):
    '''    
    For a lot size of 100 shares and tick of 0.01, an order of 120 shares will have the last 20 executed at price + 0.01
    Attributes:
        stock (Stock): the only stock on this exchange
        lot (int): the lot size 
        tick (float): the tick measured in USD
        impact (float): must be between 0 (all temporary impact) and 1 (full permanent impact)
        num_shares_owned (int): the position the agent has at this exchange
        max_holding (int): the max number of shares the agent can long or short in cumulative position
    '''
    
    def __init__(self, stock, lot=100, tick=0.1, impact=0, max_holding=100000):
        if impact < 0 or impact > 1:
            raise ValueError('impact must be a float between 0 and 1')
        self.stock = stock
        self.lot = lot
        self.tick = tick        
        self.impact = impact
        self.max_holding = max_holding
        self.num_shares_owned = 0
    
    def execute(self, order):
        ''' Execute the order, set the stock price based on self.impact, then calculate transaction cost        
        Args:
            order (int): how much shares to buy (positive) or sell (negative)
        Returns:
            float: transaction cost
        '''
        
        # First execute order using stepping up of price based on lot, tick
        # When executing the order, if it pushes num_shares_owned above max_holding, only execute in part
        # Calculate total amount paid or received by the agent for this order
        # Based on self.impact, set the new stock price
        buyOrSell = order / abs(order)
        #if it pushes num_shares_owned above max_holding, only execute in part
        if (self.num_shares_owned + order > self.max_holding):
            order =  self.max_holding - self.num_shares_owned
        if (self.num_shares_owned + order < -self.max_holding):
            order = -self.max_holding - self.num_shares_owned
        transactionCost = 0
        sharesLeft = abs(order)
        amountPaid = 0
        while sharesLeft > 0:
            amountPaid = amountPaid + self.stock.get_price() * min(lot, sharesLeft) * buyOrSell
            sharesLeft = sharesLeft - min(lot, sharesLeft)
            #update transaction cose
            transactionCost = min(lot, sharesLeft) * buyOrSell * tick + (min(lot, sharesLeft) * buyOrSell)**2 / lot * tick
            #update price 
            self.stock.set_price(self.stock.get_price + min(sharesLeft/lot, 1) * tick * buyOrSell)
        # Update num_shares_owned
        self.num_shares_owned = min(self.num_shares_owned + order, self.max_holding)
        
        return  transactionCost

    def simulate_stock_price(self, dt=1.0):
        '''
        Args:
            dt (float): length of time step
        Returns:
            float: the new share price
            float: one step pnl based on num_shares_owned and change in stock price
        '''
        
        # call stock.simulate_price(dt) 
        old_price = self.stock.get_price()
        new_price = self.stock.simulate_price(dt)
        return new_price, num_shares_owned * (new_price - old_price)
        
        
