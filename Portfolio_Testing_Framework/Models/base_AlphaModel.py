# base model
class Base_AlphaModel:
    def __init__(self, bt, **param): # bt -> reference to backtrader 
        self.signals = {}
        self.bt = bt
        pass
    
    def addSignal(self, asset, direction):
        self.signals[asset] = { 'direction': direction }

    def resetSignals(self):
        self.signals = {}
        
    def get_signals(self, assets=None, data=None):  # assets -> assets dictionory
        if(assets is not None):                     # data -> input pandas dataframe
            for k in assets.keys():
                self.addSignal(k, None)
        return self.signals                        # otput is a signal dictionary:
                                                   # example: {'BKNG': {'direction': 1}, 'AMZN': {'direction': -1}}
                                                   # direction: 1 - long, -1 - short
