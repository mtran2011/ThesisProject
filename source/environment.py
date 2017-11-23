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
    # Override base class abstractmethod
    def run(self, util, nrun, report=False):
        # reset last_action and last_state of learner to None
        # so that it doesn't use the initial reward=0 to learn internally
        self.learner.reset_last_action()

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
    # Override base class abstractmethod
    def run(self, util, nrun, report=False):
        # reset last_action and last_state of learner to None
        # so that it doesn't use the initial reward=0 to learn internally
        self.learner.reset_last_action()

        reward = 0
        # state = (stock price, option portfolio price, current share holding)
        state = (self.exchange.get_stock_price(), self.exchange.get_option_price(), 0)        
        deltas, share_holdings = [], []
        for iter_count in range(1,nrun+1):
            # order should aim for a total position close to current delta
            order = self.learner.learn(reward, state)
            # transaction_cost includes spread, impact, and change in option value
            transaction_cost = self.exchange.execute(order)
            if report:
                # compare current delta and the holdings the agent aims at
                # the holding in option is constant at max_holding so remember to scale delta with share_holdings
                deltas.append(self.exchange.get_option_delta())                
                share_holdings.append(self.exchange.num_shares_owned / self.exchange.max_holding)
            
            # after order is executed and the agent had aimed for delta, now the stock moves
            # the exchange simulates the stock and calculate pnl from both stock AND option
            new_stock_price, pnl = self.exchange.simulate_stock_price()

            # if after 1 step simulation, option.tau = -1, the pnl above is invalid, the reward is invalid
            # need to reset and do not use the invalid reward for internal learner training
            if self.exchange.check_option_expired():
                pnl = 0
                self.exchange.new_option_episode()
                self.learner.reset_last_action()

            delta_wealth = pnl - transaction_cost            
            reward = - delta_wealth**2
            state = (new_stock_price, self.exchange.get_option_price(), self.exchange.num_shares_owned)

            if iter_count % 20000 == 0:
                print('finished {:,} runs'.format(iter_count))
        
        if report:
            return deltas, share_holdings            
        else:
            return None