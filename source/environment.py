import abc
from learner import Learner
from exchange import StockExchange

class Environment(abc.ABC):
    ''' An environment that can run an agent
    Attributes:        
        learner (Learner): the agent
        exchange (StockExchange): the exchange
    '''    

    def __init__(self, learner: Learner, exchange: StockExchange):
        self.learner = learner
        self.exchange = exchange
    
    @abc.abstractmethod
    def run(self, util, nrun, report=False):
        ''' Run an agent for nrun iterations
        Args:
            util (float): the constant in the utility function
            nrun (int): number of iterations to run
            report (boolean): True to return a list of performance over time
        Returns:
            list: list of performance, e.g., cumulative wealth
        '''
        pass

class StockTradingEnvironment(Environment):
    '''
    Attributes:        
        learner (Learner): the agent
        exchange (StockExchange): the exchange for stock trading only
    '''
    # Override base class abstractmethod
    def run(self, util, nrun, report=False):
        # if this is first run, reset last_action and last_state of learner to None
        # so that it doesn't use the initial reward=0 to learn internally
        self.learner.reset_last_action()
        self.exchange.reset_episode()

        reward = 0
        state = (self.exchange.report_stock_price(), self.exchange.num_shares_owned)
        wealth, wealths = 0, []

        for iter_ct in range(1,nrun+1):
            order = self.learner.learn(reward, state)
            transaction_cost = self.exchange.execute(order)
            pnl = self.exchange.simulate_stock_price()

            delta_wealth = pnl - transaction_cost
            wealth += delta_wealth

            reward = delta_wealth - 0.5 * util * (delta_wealth - wealth / iter_ct)**2
            state = (self.exchange.report_stock_price(), self.exchange.num_shares_owned)

            if report:
                wealths.append(wealth)
            if iter_ct % 50000 == 0:
                print('finished {:,} runs'.format(iter_ct))
        
        if report:
            return wealths
        else:
            return None

class ThreeFeatureOptionHedging(Environment):
    ''' Each state has 3 features (stock price, option price, num shares owned)
    Attributes:
        learner (Learner): the agent
        exchange (OptionHedgingExchange): the exchange for option hedging only
    '''
    # Override base class abstractmethod
    def run(self, util, nrun, report=False):
        self.learner.reset_last_action()
        self.exchange.reset_episode()

        reward = 0        
        state = (
            self.exchange.report_stock_price(), 
            self.exchange.report_option_price(), 
            self.exchange.num_shares_owned)
        deltas, scaled_share_holdings = [], []
        
        for iter_ct in range(1,nrun+1):
            # order should aim for a total position close to current delta
            order = self.learner.learn(reward, state)            
            transaction_cost = self.exchange.execute(order)
            
            if report:
                # compare current delta and the holdings the agent aims at, on the same scale 0-1
                deltas.append(self.exchange.report_option_delta())
                scaled_share_holdings.append(self.exchange.num_shares_owned / self.exchange.num_options)
            
            # after order is executed and the agent had aimed for delta, now the stock moves
            # the exchange simulates the stock and calculate pnl from both stock AND option
            pnl = self.exchange.simulate_stock_price()

            # if after 1 step simulation, option.tau = -1, the pnl above is invalid, the reward is invalid
            # need to reset and do not use the invalid reward for internal learner training
            if self.exchange.check_option_expired():
                self.learner.reset_last_action()
                self.exchange.reset_episode()
            
            reward = -(pnl - transaction_cost)**2
            state = (
                self.exchange.report_stock_price(), 
                self.exchange.report_option_price(), 
                self.exchange.num_shares_owned)

            if iter_ct % 10000 == 0:
                print('finished {:,} runs'.format(iter_ct))
        
        if report:
            return deltas, scaled_share_holdings        
        else:
            return None

class TwoFeatureOptionHedging(Environment):
    ''' Try option hedging with state consisting of only two features
    Each state is (value of option portfolio, value of stock holdings)
    '''
    # Override base class abstractmethod
    def run(self, util, nrun, report=False):
        self.learner.reset_last_action()
        self.exchange.reset_episode()

        reward = 0
        state = (
            self.exchange.report_stock_price() * self.exchange.num_shares_owned, 
            self.exchange.report_option_price() * self.exchange.num_options)
        deltas, scaled_share_holdings = [], []
        
        for iter_ct in range(1,nrun+1):
            # order should aim for a total position close to current delta
            order = self.learner.learn(reward, state)            
            transaction_cost = self.exchange.execute(order)
            
            if report:
                # compare current delta and the holdings the agent aims at, on the same scale 0-1
                deltas.append(self.exchange.report_option_delta())
                scaled_share_holdings.append(self.exchange.num_shares_owned / self.exchange.num_options)
            
            # after order is executed and the agent had aimed for delta, now the stock moves
            # the exchange simulates the stock and calculate pnl from both stock AND option
            pnl = self.exchange.simulate_stock_price()

            # if after 1 step simulation, option.tau = -1, the pnl above is invalid, the reward is invalid
            # need to reset and do not use the invalid reward for internal learner training
            if self.exchange.check_option_expired():
                self.learner.reset_last_action()
                self.exchange.reset_episode()
            
            reward = -(pnl - transaction_cost)**2
            state = (
                self.exchange.report_stock_price() * self.exchange.num_shares_owned, 
                self.exchange.report_option_price() * self.exchange.num_options)

            if iter_ct % 1000 == 0:
                print('finished {:,} runs'.format(iter_ct))
        
        if report:
            return deltas, scaled_share_holdings
        else:
            return None

class GammaScalpingEnvironment(Environment):
    ''' Try to make money if the option is underpriced using too low implied vol
    Each state is (option price, stock price, num shares owned)
    '''
    # Override base class abstractmethod
    def run(self, util, nrun, report=False):
        self.learner.reset_last_action()
        self.exchange.reset_episode()

        reward = 0
        state = (
            self.exchange.report_stock_price(),
            self.exchange.report_option_price(),
            self.exchange.report_option_tau(),
            self.exchange.num_shares_owned)
        wealths, wealth = [], 0

        for iter_ct in range(1,nrun+1):            
            order = self.learner.learn(reward, state)            
            transaction_cost = self.exchange.execute(order)
            pnl = self.exchange.simulate_stock_price()
            # if after 1 step simulation, option.tau = -1, the pnl above is invalid, the reward is invalid
            # need to reset and do not use the invalid reward for internal learner training
            if self.exchange.check_option_expired():
                self.learner.reset_last_action()
                self.exchange.reset_episode()
                pnl, transaction_cost = 0, 0
            
            delta_wealth = pnl - transaction_cost
            wealth += delta_wealth

            reward = delta_wealth - 0.5 * util * (delta_wealth - wealth / iter_ct)**2
            state = (
                self.exchange.report_stock_price(),
                self.exchange.report_option_price(),
                self.exchange.report_option_tau(),
                self.exchange.num_shares_owned)

            if report:
                wealths.append(wealth)
            if iter_ct % 1000 == 0:
                print('finished {:,} runs'.format(iter_ct))
        
        if report:
            return wealths
        else:
            return None