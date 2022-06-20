import numpy as np
from base_RebalanceModel import Base_RebalanceModel


def calcDiffAmount(current, target):
    if (current <= 0) and (target <= 0):
        d = -1 * (abs(target) - abs(current))

    if (current >= 0) and (target >= 0):
        d = (abs(target) - abs(current))

    if (current >= 0) and (target <= 0):
        d = -1 * (abs(target) + abs(current))

    if (current <= 0) and (target >= 0):
        d = (abs(target) + abs(current))

    return d

def getNewAllocationSize(cash, open_positions, new_allocation, data):

    ret = {}

    for k,v in new_allocation.items():
        cash_need = v * cash
        #print('{}:[{}] -> cash_needed: {}'.format(k, v,cash_need))
        new_size = (cash_need / data[k].iloc[-1]).astype(int)
#        print('{} current price {} -> new size: {}'.format(k, data[k].iloc[-1],new_size))
        if(k in open_positions.keys()):
            new_size = calcDiffAmount(open_positions[k]['size'], new_size)
        ret[k] = new_size

    return ret

class RebalanceModel_Simple(Base_RebalanceModel):
    def __init__(self, bt, **params):
        super().__init__(bt, **params)
        pass

    def get_allocations(self, allocation = None, data = None, **params):
        if((allocation is not None) and (data is not None)):
            self.reset_allocations()

            openPos = self.bt.getOpenPositions()
            #print('curret open:', openPos)

            cash = self.bt.getAllCash()
            #print('all_cash_pnl', cash)

            if(cash > self.bt.reserveCash):
                cash -= self.bt.reserveCash
            else:
                cash = 0.0

            to_remove = list(set(openPos.keys()) - set(allocation.keys()))

            for k in to_remove:
                allocation[k] = 0.0
            
            tm = getNewAllocationSize(cash, openPos, allocation, data)
            return tm

        else:
            return {}
