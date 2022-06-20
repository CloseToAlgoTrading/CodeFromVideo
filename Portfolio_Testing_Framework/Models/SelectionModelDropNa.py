from base_SelectionModel import Base_SelectionModel

dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

class SelectionModel_DropNa(Base_SelectionModel):

    def __init__(self, bt, assets, **params):
        super().__init__(bt, assets)
        pass

    def get_assets(self, data=None):
        if(data is not None):
            df = data.dropna(axis=1)
            df = df.drop(columns=df.columns[(df == 0.0).any()])
            ret = dict_filter(self.assets, df.columns.values)
            return ret
        else:
            return self.assets

    
