# base model
class Base_RiskModel:
    def __init__(self, bt, **params):   # bt -> reference to backtrader
        self.bt = bt
        pass
        
    def get_allocations(self, allocations = None, data = None, **params):# allocation -> allocation dictionary
        return allocations                                               # data -> historical data pandas data frame
                                                                         # ouput is an allocation dictionary in value
                                                                         # example: {'BKNG': 0, 'AMZN': -35}

        