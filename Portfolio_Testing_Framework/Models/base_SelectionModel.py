# base model
class Base_SelectionModel:
    def __init__(self, bt, assets):  # <- bt -> reference to backtrader
        self.assets = assets         # assets -> asset dictionory 
        self.bt = bt
        pass

    def get_assets(self, data=None):  # <--- Input pandas data frame
        return self.assets            # <--- asset dictionory 
                                      #      key   -> ticker
                                      #      value -> data id in backtrader datas
                                      #      example: {'A': 1, 'AAP': 2, 'ABMD': 4}
