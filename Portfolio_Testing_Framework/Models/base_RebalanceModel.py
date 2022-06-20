# base model
class Base_RebalanceModel:
    def __init__(self, bt, **params):  # bt -> reference to backtrader
        self.allocations = {}
        self.prevPercentage = {}
        self.bt = bt
        pass

    def reset_allocations(self):
        self.allocations = {}
        
    def get_allocations(self, allocation = None, data = None, **params): # allocation -> allocation dictionary
        return self.allocations                                          # data -> historical data pandas data frame
                                                                         # ouput is an allocation dictionary in value
                                                                         # example: {'BKNG': 40, 'AMZN': -35}