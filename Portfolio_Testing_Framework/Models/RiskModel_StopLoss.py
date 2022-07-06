from base_riskModel import Base_RiskModel

class RiskModel_StopLoss(Base_RiskModel):
    def __init__(self, bt, **param):
        super().__init__(bt, **param)
        self.stopLoss = param.get('stopLoss', 0.0)
        self.resetRebalanceCounter = param.get('resetRebalanceCounter', False)
        pass

    def get_allocations(self, allocations = None, data = None, **params):

        openPos = self.bt.getOpenPositions()
        
        if(self.stopLoss > 0) and (self.stopLoss < 1.0) and (len(openPos) > 0) and ((allocations is None) or (len(allocations) == 0)):
            
            ininital_dollar_amount = 0
            pnl = 0
            for k,v in openPos.items():
                ininital_dollar_amount += abs(v['openPrice'] * v['size'])
                pnl += (v['currentPrice'] - v['openPrice']) * v['size'] 

            pnl = pnl / ininital_dollar_amount

            if(pnl < -self.stopLoss):
                for k,v in openPos.items():
                    allocations[k] = -v['size']
                
                if (True == self.resetRebalanceCounter):
                    self.bt.update_counter = 1



        return allocations
