from base_AlphaModel import Base_AlphaModel

class AlphaModel_GenericMomentum(Base_AlphaModel):
    def __init__(self, bt, **param):
        super().__init__(bt, **param)
        self.momentum_window = param.get('momentum_window', 252)
        self.n_top = param.get('n_top', 10)
        pass

    def get_signals(self, assets=None, data=None):
        
        self.resetSignals()

        if((assets is not None) and (data is not None)):
            df = data.loc[:,list(assets)]

            #calculate momentum
            momentum = (df.iloc[-1] / df.iloc[-self.momentum_window] - 1)
            nTop = momentum.nlargest(self.n_top).index.values
            for a in nTop:
                self.addSignal(a, direction=1)
            
        return self.signals
