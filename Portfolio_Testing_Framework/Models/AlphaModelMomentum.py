from base_AlphaModel import Base_AlphaModel
from helper_lib import calculateFIP, calculateMomentum, calculateMaMomentum

class AlphaModel_Momentum(Base_AlphaModel):
    def __init__(self, bt, **param):
        super().__init__(bt, **param)
        self.momentum_window = param.get('momentum_window', 252)
        self.n_top = param.get('n_top', 10)
        self.n_bot = param.get('n_bot', 10)
        self.isFIPUsed = param.get('isFIPUsed', False)
        self.n_FIP = param.get('n_FIP', 25)
        self.isShortActive= param.get('isShortActive', False)
        self.isLongActive= param.get('isLongActive', False)
        self.isInvertMomentunUsed = param.get('invertMomentum', False)
        
        self.momentums =  {
            'generic_momentum':calculateMomentum,
            'ma_momentum' : calculateMaMomentum
        }

        self.momentum_function = self.momentums.get(param.get('momentum_type', 'generic_momentum'), calculateMomentum)

        pass

    def selectBasedOnFip(self, momentum, prices, isLong, n):
        if True == isLong: 
            nAssets = momentum.nlargest(self.n_FIP).index.values
        else:
            nAssets = momentum.nsmallest(self.n_FIP).index.values
        
        k = calculateFIP(prices.loc[:,nAssets].iloc[-self.momentum_window:])
        return k.nsmallest(n).index.values

    def get_signals(self, assets=None, data=None):
        
        self.resetSignals()

        if((assets is not None) and (data is not None)):
            df = data.loc[:,list(assets)]
            nTop = []
            nBot = []

            #calculate momentum
            momentum = self.momentum_function(df, -1, self.momentum_window)
            if ( True == self.isInvertMomentunUsed ):
                momentum *= -1.0
                
            # filters
            if self.isFIPUsed == True:
                if(True == self.isLongActive):
                    nTop = self.selectBasedOnFip(momentum, df, True, self.n_top)    
                if(True == self.isShortActive):    
                    nBot = self.selectBasedOnFip(momentum, df, False, self.n_bot)  
            else:
                if(True == self.isLongActive):    
                    nTop = momentum.nlargest(self.n_top).index.values
                if(True == self.isShortActive):    
                    nBot = momentum.nsmallest(self.n_bot).index.values

            # create signals
            if(True == self.isLongActive):
                for a in nTop:
                    self.addSignal(a, direction=1)
            
            if(True == self.isShortActive):
                for a in nBot:
                    self.addSignal(a, direction=-1)


        return self.signals
