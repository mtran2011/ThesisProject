import abc 

class StockExchange(object):
    '''
    Attributes:
        stock (Stock): the single stock on this exchange
        order_size_limit (int): maximum number of shares in an order
        
    '''
    
    def publish_asset_price(self):
        return self.stock.price
        
    def execute_order(self):