import itertools
import matplotlib.pyplot as plt
from stock import OUStock, GBMStock
from exchange import StockExchange, StockOptionExchange
from qlearner import QMatrix, QMatrixHeuristic, SemiGradQLearner, DQNLearner
from environment import StockTradingEnvironment, OptionHedgingEnvironment
from function_estimator import PairwiseLinearEstimator
from option import EuropeanStockOption
import distance_func
# import model_builder

def graph_performance(wealths_list, agent_names, ntrain):
    linestyles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    
    plt.figure()
    for wealths, agent in zip(wealths_list, agent_names):
        plt.plot(range(len(wealths)), wealths, label=agent, 
                 linestyle=next(linestyles),
                 color=next(colors))
    
    ntest = len(wealths_list[0])
    plt.title('Performance with ntrain = {0:,} and ntest = {1:,}'.format(ntrain, ntest))
    plt.legend(loc='best')
    plt.xlabel('iterations of testing runs')
    plt.ylabel('cumulative wealth')
    plt.savefig('../figs/newfig.png')

def make_stock_exchange():
    stock = OUStock(price=10, kappa=0.1, mu=12, sigma=0.2, tick=0.1, band=1000)
    lot = 10
    actions = list(range(-3*lot, 4*lot, lot))
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=100)
    return actions, exchange

def make_option_exchange():
    stock = GBMStock(price=10, mu=0.005, sigma=0.01, tick=0.1, band=1000)
    option = EuropeanStockOption(stock, k=10, tau=(1e6+5010), r=0.001, is_call=True)
    lot = 10
    actions = list(range(-5*lot, 5*lot, lot))
    exchange = StockOptionExchange(option, lot=lot, impact=0, max_holding=5*lot)
    return actions, exchange

def run_qmatrix_option_hedging():
    actions, exchange = make_option_exchange()
    util, ntrain, ntest = 1e-3, int(1e6), 5000

    # for QMatrixHeuristic
    dist_func = lambda x1, x2: distance_func.p_norm(x1, x2, p=2)
    qheuristic_learner = QMatrixHeuristic(actions, dist_func, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    environment = OptionHedgingEnvironment(qheuristic_learner, exchange)
    environment.run(util, ntrain)
    wealths_qheuristic, deltas, share_holdings = environment.run(util, ntest, report=True)

    graph_performance([deltas, share_holdings], ['option delta', 'scaled share holding'], ntrain)

def run_qmatrix_stock_trading():
    actions, exchange = make_stock_exchange()
    util, ntrain, ntest = 1e-3, int(1e4), 5000
    
    # for SemiGradQLearner    
    qfunc_estimator = PairwiseLinearEstimator(num_state_features=2)
    qgrad_learner = SemiGradQLearner(actions, qfunc_estimator, epsilon=0.1, learning_rate=1e-5, discount_factor=0.999)
    environment = StockTradingEnvironment(qgrad_learner, exchange)
    environment.run(util, ntrain)    
    wealths_semigrad = environment.run(util, ntest, report=True)
    print(qfunc_estimator.get_params())

    # for simple QMatrix
    qmatrix_learner = QMatrix(actions, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    environment = StockTradingEnvironment(qmatrix_learner, exchange)
    environment.run(util, ntrain)
    wealths_qmatrix = environment.run(util, ntest, report=True)

    # for QMatrixHeuristic
    dist_func = lambda x1, x2: distance_func.p_norm(x1, x2, p=2)
    qheuristic_learner = QMatrixHeuristic(actions, dist_func, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    environment = StockTradingEnvironment(qheuristic_learner, exchange)
    environment.run(util, ntrain)
    wealths_qheuristic = environment.run(util, ntest, report=True)

    graph_performance([wealths_qmatrix, wealths_qheuristic, wealths_semigrad],
                      ['discrete Q matrix', 'heuristic Q matrix', 'semigrad Q learner'], ntrain)    

def run_dqn_stock_trading():
    actions, exchange = make_stock_exchange()
    util, ntrain, ntest = 1e-3, int(1e6), 5000

    model = model_builder.build_simple_ff(len(actions), 2, len(actions))
    learner = DQNLearner(actions, model, epsilon=0.1, discount_factor=0.999)
    environment = StockTradingEnvironment(learner, exchange)

    environment.run(util, ntrain)    
    wealths = environment.run(util, ntest, report=True)
    graph_performance([wealths], ['simple_dqn_feed_forward'], ntrain)    

if __name__ == '__main__':
    run_qmatrix_option_hedging()