import numpy as np
from base_AllocationModel import Base_AllocationModel
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from pypfopt import HRPOpt, CLA, EfficientCVaR, EfficientCDaR
from pypfopt import objective_functions

class AllocationModel_PyFo(Base_AllocationModel):
    def __init__(self, bt, **params):
        super().__init__(bt, **params)
        self.method = params.get('method', 'HRP')
        pass

    def get_allocations(self, signals = None, data = None, **params):
        if((signals is not None) and (data is not None)):
            self.resetAllocations()

            signal_counts = 0
            assets = []
            for k,v in signals.items():
                if((v['direction'] == 1) or (v['direction'] == -1)):
                    self.allocation[k] = 0.0
                    signal_counts += 1
                    assets.append(k)
            if signal_counts > 0:
                prices = data.loc[:,assets]

                if('HRP' == self.method):
                    opt = self.HRP(prices)
                elif('CVaR' == self.method):                    
                    opt = self.efCVaR(prices)
                elif('CDaR' == self.method):                    
                    opt = self.efCDaR(prices)
                elif('CLA' == self.method):                    
                    opt = self.CLA(prices)
                elif('E_Frontier' == self.method):
                    opt = self.ef(prices)
   
                cleaned_weights = opt.clean_weights()
                #print(cleaned_weights)
                

                for a,v in cleaned_weights.items():
                    self.allocation[a] = v

            return self.allocation

        else:
            return {}


    def efCVaR(self, data):
        returns = expected_returns.returns_from_prices(data).dropna()
        mu = expected_returns.capm_return(data)
        ec = EfficientCVaR(mu, returns)
        ec.min_cvar()
        #ec.efficient_risk(0.02, market_neutral=True)
        return ec

    def efCDaR(self, data):
        returns = expected_returns.returns_from_prices(data).dropna()
        mu = expected_returns.capm_return(data)
        ec = EfficientCDaR(mu, returns)
        ec.min_cdar()
        #ec.efficient_risk(0.02, market_neutral=True)
        return ec

    def ef(self, data):
        exp_ret = expected_returns.capm_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        opt = EfficientFrontier(exp_ret, S, weight_bounds=(-1, 1))
        opt.add_objective(objective_functions.L2_reg)
        #opt.efficient_risk(target_volatility=0.05, market_neutral=False)
        opt.min_volatility()      
        return opt

    def HRP(self, data):
        exp_ret = expected_returns.returns_from_prices(data)
        opt = HRPOpt(exp_ret)
        opt.optimize()        
        return opt

    def CLA(self, data):
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        mu = expected_returns.capm_return(data)
        opt = CLA(mu, S)
        opt.min_volatility()
        return opt