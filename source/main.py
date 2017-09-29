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