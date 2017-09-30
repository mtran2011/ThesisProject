from stock import OUStock

def main():
    '''
    initialize new ou stock(price=10, *lambda)  
    action_space = [-100,0,100,200]
    
    show agent state=(price=10, owned=0)
    
    agent loops over action_space:
        based on state=(price=10, owned=0), find highest q, and thus best action 
        return action = buy 200
    
    environment:
        execute order to buy 200, bump price which means transaction cost of 0.01
        simulate price from 10 to 11
        calculate reward = 1 + transaction cost = 0.99
        give agent reward, new state=(price=9, owned=200)
        
    agent update q using reward=0.99, newstate      
    '''
    stock = OUStock(10.52, kappa=0.1, mu=16.0, sigma=0.3/252)
    lot, tick = 100, 0.01
    actions = range(-10*lot, 11*lot, lot)
    qlearner = QLearner(actions)    
    exchange = StockExchange(stock, lot, tick, impact=0)
    reward = 0
    state = (stock.get_price(), shares_owned=0)
    for _ in range(1000000):
        order = qlearner.learn(reward, state)
        exchange.execute(order) 
        exchange.update_position_count()
        exchange.report_transaction_cost()        
        exchange.simulate_stock_price()
        reward = exchange.calculate_pnl()
        state = (exchange.report_stock_price(), exchange.report_position_count())
        
    
    
if __name__ == '__main__':
    main()
    print('all good')