import abc

class Environment(abc.ABC):
    ''' An environment that can run an agent
    Attributes:        
        learner (QLearner): the agent
        exchange (StockExchange): the exchange
    '''    

    def __init__(self, learner, exchange):
        self.learner = learner
        self.exchange = exchange
    
    @abc.abstractmethod
    def run(self, util, nrun, report=False):
        ''' Run an agent for nrun iterations
        Args:
            util (float): the constant in the utility function
            nrun (int): number of iterations to run
            report (boolean): True to return a list of cumulative_wealth over time
        Returns:
            list: list of cumulative wealth
        '''
        pass

class StockTradingEnvironment(Environment):
        
    def run(self, util, nrun, report=False):        
        reward = 0
        # state = (stock price, current share holding)
        state = (self.exchange.get_stock_price(), 0)
        iter_count, cumulative_wealth = 0, 0         
        wealths = []
        
        while iter_count < nrun:
            order = self.learner.learn(reward, state)
            transaction_cost = self.exchange.execute(order)
            new_stock_price, pnl = self.exchange.simulate_stock_price()
            
            delta_wealth = pnl - transaction_cost        
            cumulative_wealth += delta_wealth
            iter_count += 1
            
            reward = delta_wealth - util / 2 * (delta_wealth - cumulative_wealth / iter_count)**2
            state = (new_stock_price, self.exchange.num_shares_owned)
            
            if report:
                wealths.append(cumulative_wealth)
            if iter_count % 20000 == 0:
                print('finished {:,} runs'.format(iter_count))
        
        if report:
            return wealths
        else:
            return None

class OptionHedgingEnvironment(Environment):
        
    def run(self, util, nrun, report=False):
        reward = 0
        # state = (single stock price, price of option portfolio, current share holding)
        state = (self.exchange.get_stock_price(), self.exchange.get_option_price(), 0)
        iter_count, cumulative_wealth = 0, 0
        wealths, deltas, share_holdings = [], [], []
        while iter_count < nrun:
            order = self.learner.learn(reward, state)
            # transaction_cost includes spread, impact, and change in option value
            transaction_cost = self.exchange.execute(order)
            # the exchange simulates the stock and calculate pnl from both stock and option
            new_stock_price, pnl = self.exchange.simulate_stock_price()

            delta_wealth = pnl - transaction_cost        
            cumulative_wealth += delta_wealth
            iter_count += 1
            
            reward = delta_wealth - util / 2 * (delta_wealth - cumulative_wealth / iter_count)**2
            state = (new_stock_price, self.exchange.get_option_price(), self.exchange.num_shares_owned)

            if report:
                wealths.append(cumulative_wealth)
                # the holding in option is constant at max_holding so have to scale share_holdings
                deltas.append(self.exchange.option.find_delta())
                share_holdings.append(self.exchange.num_shares_owned / self.exchange.max_holding)
            if iter_count % 20000 == 0:
                print('finished {:,} runs'.format(iter_count))
        
        if report:
            return wealths, deltas, share_holdings
        else:
            return None