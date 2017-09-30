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
    '''
    
    def __init__(self, stock, lot=100, tick=0.1, impact=0):
        if impact < 0 or impact > 1:
            raise ValueError('impact must be a float between 0 and 1')
        self.stock = stock
        self.lot = lot
        self.tick = tick
        self.impact = impact        
        self.num_shares_owned = 0
    
    def execute(order):
        ''' Execute the order, set the stock price based on self.impact, then calculate transaction cost        
        Args:
            order (int): how much shares to buy (positive) or sell (negative)
        Returns:
            float: transaction cost
        '''
        
        # First execute order using stepping up of price based on lot, tick
        # Update num_shares_owned
        # Calculate total amount paid or received by the agent for this order
        
        # Based on self.impact, set the new stock price
        
        # Return transaction cost based on updated stock price and total amount paid

    def simulate_stock_price(dt=1.0):
        '''
        Args:
            dt (float): length of time step
        Returns:
            float: the new share price
            float: one step pnl based on num_shares_owned and change in stock price
        '''
        
        # call stock.simulate_price(dt) 
        new_price = 
        
        