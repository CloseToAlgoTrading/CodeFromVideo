import numpy as np
# base model
class Base_AllocationModel:
    def __init__(self, bt, **params):     # bt -> reference to backtrader
        self.allocation = {}
        self.bt = bt
        pass
    def resetAllocations(self):
        self.allocation = {}
        pass
    
    def get_allocations(self, signals = None, data = None, **params):  # signals -> signals dictionary
        return self.allocation                                         # data -> historical data pandas data frame
                                                                       # ouput is an allocation dictionary in percentage
                                                                       # example: {'BKNG': 0.1, 'AMZN': -0.1}
                                                                       

