'''
class Environment:
    To manage the back and forth between agent-learner and an exchange. First the exchange reports a bunch of info on price, etc. 
    Use this info to calculate a reward r. Combine these info into a tuple or a state s. Show (r,s) to agent. 
    Agent gives back an action/order. Pass this order to exchange to execute, get transaction cost from it. Form a new (r,s) and so on

    For the option case, an episode finishes when the option expires. You need to restart a new episode.
    So, if the exchange reports that the option has expired, environment has to reset by calling learner.reset() and exchange.reset()

    Attributes:
        learner: the agent
        exchange: the exchange
    Methods:
        run(nrun): run the above for nrun iterations


class Stock:
    Attributes:
        mu, sigma: input constant parameters
        tick: 0.01 for 1 cent, 0.1 for 10 cent, or 1 dollar; price can only move in multiple of ticks
        price: float rounded off up to 2 decimals, based on ticks
    Methods:
        simulate_price(dt): simulate its own price for a time step dt
        get_price(): report rounded price
        set_price(val): for the exchange to set a new price based on market impact of order

class Option:
    Attributes:
        stock: pointer to underlying stock
        strike, expiry (original time to expiry), tau (remaining time to expiry), rate (risk free rate)
        price, delta: these are calculated inside this class
    Methods:
        get_price(): report rounded price based on stock.tick
        update_price(): do the calculation to update price, delta

class StockExchange:
    Mainly does 3 things: 
        keep track of num_shares_owned, how many shares the agent has right now; 
        execute order of buying or selling 100, or 1500 shares, and calculate transaction cost including market impact
        simulate stock price by 1 step and calculate resulting PnL based on new price and num_shares_owned

    Attributes:
        stock: pointer to stock
        num_shares_owned: how many shares the agent owns now
        lot, impact: constant inputs for lot size, impact factor
        max_holding: the agent can max long or max short this number of shares
    Methods:
        get_stock_price(): report stock price to the trading environment
        execute(order): execute a buy or sell order of shares, update num_shares_owned, return transaction cost
        simulate_stock_price(dt): call stock.simulate_price(dt); return the PnL from 1 time step move

class StockOptionExchange(StockExchange):
    Inherits above. But when it simulates the stock, the PnL = PnL in stock + PnL from a portfolio of options
    Important: we assume the number of options held is always fixed at exchange.max_holding

    Attributes:
        option: added field for the underlying option
    Methods:
        simulate_stock_price(dt): simulates the stock, decrement option.tau by dt, reprice the option, return combined PnL

class QLearner:
    This is the Q-learning agent


    





def main():
    stock = Stock(*args)
    option = option(stock, *args)
    exchange = OptionExchange(option, *args)

    kernel = KernelSmoother(*args)
    list_possible_actions = list(range(-100,100))
    learner = KernelQLearner(kernel, list_possible_actions, *args)

'''