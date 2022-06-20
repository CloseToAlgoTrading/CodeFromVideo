# base model
class Base_ExecutionModel:
    def __init__(self, bt, **params):  # bt -> reference to back trader
        self.bt = bt
        pass
       
    def execute(self, allocation = None, data = None, **params): # allocation - allocation dictionary
                                                                 # data - historical data (pandas data frame)
        to_sell = []
        to_buy = []
        for k,v in allocation.items():
            if(v<0):
                to_sell.append(k)
            elif(v>0):
                to_buy.append(k)

        # first sell assets to free cash
        for k in to_sell:
            self.bt.sell(k, allocation[k])
        # buy assets
        for k in to_buy:
            self.bt.buy(k, allocation[k])

        pass
