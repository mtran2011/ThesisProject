import abc

class Environment(metaclass=abc.ABCMeta):
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
            nrun (int): number of iterations
            report (boolean): True to return a list of cumulative_wealth over time
        Returns:
            list: list of cumulative wealth
        '''
        pass

class StockTradingEnvironment(Environment):
        
    def run(self, util, nrun, report=False):        
        reward = 0
        state = (self.exchange.stock.get_price(), 0)
        iter_count, cumulative_wealth = 0, 0         
        wealths = []
        
        while iter_count < nrun:
            order = self.learner.learn(reward, state)
            # when the exchange execute, it makes an impact on stock price
            transaction_cost = self.exchange.execute(order)        
            new_price, pnl = self.exchange.simulate_stock_price()
            
            delta_wealth = pnl - transaction_cost        
            cumulative_wealth += delta_wealth
            iter_count += 1
            
            reward = delta_wealth - util / 2 * (delta_wealth - cumulative_wealth / iter_count)**2
            state = (new_price, self.exchange.num_shares_owned)
            
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
        state = (self.exchange.stock.get_price(), self.exchange.option.tau, 0)
        iter_count, cumulative_wealth = 0, 0         
        wealths = []
        pass