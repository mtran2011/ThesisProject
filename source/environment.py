from stock import Stock, OUStock
from exchange import StockExchange
from qlearner import QLearner, QMatrix, DQNLearner, QMatrixHeuristic

class StockTradingEnvironment(object):
    ''' An environment that can run a stock trading agent
    Attributes:        
        learner (QLearner): the agent
        exchange (StockExchange): the exchange
    '''
    
    def __init__(self, learner, exchange):
        self.learner = learner
        self.exchange = exchange        
        
    def run(self, util, nrun, report=False):
        ''' Run a stock trading agent for nrun iterations
        Args:        
            util (float): the constant in the utility function
            nrun (int): number of iterations
            report (boolean): True to return a list of cumulative_wealth over time
        Returns:
            list: list of cumulative wealth
        '''
        reward = 0
        state = (self.exchange.stock.get_price(), 0)    
        iter_count = 0
        cumulative_wealth = 0
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
            if iter_count % 10000 == 0:
                print('finished {0} runs'.format(iter_count))
        
        if report:
            return wealths
        else:
            return None