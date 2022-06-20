import numpy as np
from base_AllocationModel import Base_AllocationModel

class AllocationModel_Equal(Base_AllocationModel):
    def __init__(self, bt, **params):
        super().__init__(bt, **params)
        pass

    def get_allocations(self, signals = None, data = None, **params):
        if((signals is not None) and (data is not None)):
            self.resetAllocations()
            w = 0
            signal_counts = 0
            for k,v in signals.items():
                if((v['direction'] == 1) or (v['direction'] == -1)):
                    self.allocation[k] = 0.0
                    signal_counts += 1
            if signal_counts > 0:
                w = 1 / signal_counts 
            for k,v in self.allocation.items():
                self.allocation[k] = w * signals[k]['direction']
            return self.allocation

        else:
            return {}
