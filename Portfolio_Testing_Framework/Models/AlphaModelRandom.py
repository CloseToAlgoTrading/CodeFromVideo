from base_AlphaModel import Base_AlphaModel
import random

class AlphaModel_Random(Base_AlphaModel):
    def __init__(self, bt, **param):
        super().__init__(bt, **param)
        self.n_top = param.get('n_top', 10)
        self.n_bot = param.get('n_bot', 10)
        self.isShortActive= param.get('isShortActive', False)
        self.isLongActive= param.get('isLongActive', False)
        self.seed= param.get('seed', None)

        if self.seed is not None:
            random.seed(self.seed)
        
        pass


    def get_signals(self, assets=None, data=None):
        
        self.resetSignals()

        if((assets is not None) and (data is not None)):
            df = data.loc[:,list(assets)]
            nTop = []
            nBot = []

            if(True == self.isLongActive):    
                nTop = random.sample(df.columns.tolist(), self.n_top)
            if(True == self.isShortActive): 
                rest_col = list(set(df.columns.tolist()) - set(nTop))   
                nBot = random.sample(rest_col, self.n_bot)

            # create signals
            if(True == self.isLongActive):
                for a in nTop:
                    self.addSignal(a, direction=1)
            
            if(True == self.isShortActive):
                for a in nBot:
                    self.addSignal(a, direction=-1)


        return self.signals
