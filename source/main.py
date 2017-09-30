from stock import OUStock
from exchange import StockExchange
from qlearner import QLearner, QMatrix

def main():    
    stock = OUStock(10.52, kappa=0.1, mu=16.0, sigma=0.3/252)
    lot, tick = 100, 0.1    
    actions = list(range(-10*lot, 11*lot, lot))
    learner = QMatrix(actions)    
    exchange = StockExchange(stock, lot, tick, impact=0)       
    
    reward = 0
    state = (stock.get_price(), 0)
    
    iter_count = 0
    cumulative_wealth = 0    
    
    # for initial training and burn in
    for i in range(1000000):
        order = learner.learn(reward, state)
        
        # when the exchange execute, it makes an impact on stock price
        transaction_cost = exchange.execute(order)        
        new_price, pnl = exchange.simulate_stock_price()
        
        delta_wealth = pnl + transaction_cost
        iter_count += 1
        cumulative_wealth += delta_wealth        
        
        reward = delta_wealth - util_const / 2 * (delta_wealth - cumulative_wealth / iter_count)**2
        state = (new_price, exchange.num_shares_owned)        
    
    # for graphing pnl after training, run again the above 10k times
    
if __name__ == '__main__':
    main()
    print('all good')