from stock import Stock
from option import Pair

class StockExchange(object):
    ''' A class to execute buy or sell order for a single stock 
    For a lot size of 100 shares and tick of 0.1, an order to buy 120 shares will have the last 20 executed at price + 0.1
    Attributes:
        stock (Stock): the only stock on this exchange
        lot (int): the lot size         
        impact (float): must be between 0 (all temporary impact) and 1 (full permanent impact)
        max_holding (int): the max number of shares the agent can long or short in cumulative position
        num_shares_owned (int): the position the agent has at this exchange        
    '''
    
    def __init__(self, stock : Stock, lot : int, impact : float, max_holding : int):
        if impact < 0 or impact > 1:
            raise ValueError('impact must be a float between 0 and 1')
        if lot < 0 or max_holding < 0:
            raise ValueError('lot and max_holding must be positive')        
        self._stock = stock        
        self.lot = lot
        self.impact = impact
        self.max_holding = max_holding
        self.num_shares_owned = 0
    
    def reset_episode(self):
        ''' Reset is like starting a new game episode
        '''
        self.num_shares_owned = 0

    def report_stock_price(self):
        ''' Return stock price that is already rounded
        '''
        return self._stock.get_price()

    def execute(self, order : int):
        ''' Execute the order, set the stock price based on self.impact, then calculate transaction cost        
        Args:
            order (int): how many shares to buy (positive) or sell (negative)
        Returns:
            float: transaction cost
        '''
        
        # First execute order using stepping up of price based on lot, tick        
        # Calculate total amount paid or received by the agent for this order        
        if order == 0:
            return 0
        if not isinstance(order, int):
            raise ValueError('order must be type int')
        
        buy_or_sell = order / abs(order)
        
        # When executing the order, if it pushes num_shares_owned above max_holding, only execute in part
        if self.num_shares_owned + order > self.max_holding:
            order =  self.max_holding - self.num_shares_owned
        if self.num_shares_owned + order < -self.max_holding:
            order = -self.max_holding - self.num_shares_owned
        
        shares_left = abs(order)
        transaction_cost = 0        
        amount_paid = 0
        # the first bid or offer is 1 tick from stock.get_price() which is assumed to be a mid price
        tick = self._stock.tick
        price_to_execute = self._stock.get_price() + tick * buy_or_sell
        
        while shares_left > 0:
            shares_to_execute = min(self.lot, shares_left)        
            amount_paid += price_to_execute * shares_to_execute * buy_or_sell
            shares_left -= shares_to_execute
            price_to_execute += tick * buy_or_sell
            
            # update transaction cost
            spread_cost = shares_to_execute / self.lot * tick
            impact_cost = (shares_to_execute / self.lot)**2 * tick
            transaction_cost += spread_cost + impact_cost
        
        # update num_shares_owned
        self.num_shares_owned += order
        # Based on self.impact, set the new stock price
        self._stock.set_price(self._stock.get_price() + self.impact * (price_to_execute - self._stock.get_price()))
        
        return transaction_cost

    def simulate_stock_price(self, dt=1):
        '''
        Args:
            dt (float): length of time step
        Returns:            
            float: one step pnl based on num_shares_owned and change in stock price
        '''
        old_price = self._stock.get_price()
        new_price = self._stock.simulate_price(dt)
        return self.num_shares_owned * (new_price - old_price)

class OptionHedgingExchange(StockExchange):
    ''' Assume that the number of options held on exchange is constant and equal to max_holding
    Attributes:
        pair (Pair): a pair of option and stock
        num_options (int): number of options held, always constant, never changes
    '''
    def __init__(self, pair : Pair, lot : int, impact : float, max_holding : int):
        super().__init__(pair.get_stock(), lot, impact, max_holding)
        self._pair = pair
        self.num_options = max_holding
    
    def reset_episode(self):
        ''' Reset both the option and the stock
        '''
        # reset num_shares_owned to zero
        super().reset_episode()
        # reset option to original expiry and stock to original price
        self._pair.reset_episode()

    def report_option_price(self):
        ''' Return option price that is already rounded to tick
        '''
        return self._pair.get_option_price()
    
    def report_option_delta(self):
        ''' Return delta of the option, a float
        '''
        return self._pair.get_option_delta()
    
    def check_option_expired(self):
        ''' Return True if the option has already expired
        '''
        return self._pair.check_option_expired()
    
    def simulate_stock_price(self, dt=1):
        ''' Return one step pnl from both stock and repriced option
        Decrement option.tau by time step dt
        '''        
        old_stock_price = self.report_stock_price()
        old_option_price = self.report_option_price()
        # self._pair.simulate_stock_price()should decrement option.tau by dt before repricing it
        new_stock_price, new_option_price = self._pair.simulate_stock_price()
        pnl = self.num_shares_owned * (new_stock_price - old_stock_price) + self.num_options * (new_option_price - old_option_price)
        return pnl