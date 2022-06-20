#Following code need to be here for simple access to the modules
import sys
sys.path.insert(0, './../../common')
sys.path.insert(0, './../../Models')
sys.path.insert(0, './common')
sys.path.insert(0, './Models')


import numpy as np
from datetime import datetime
import os
import pandas as pd
import backtrader as bt
import copy


from SelectionModelDropNa import SelectionModel_DropNa
from AlphaModelGenericMomentum import AlphaModel_GenericMomentum
from AllocationModelEqual import AllocationModel_Equal
from RebalanceModelSimple import RebalanceModel_Simple



from bt_testcode3 import backtest, DynRebalance
from pandas_datareader import data as pdr
import yfinance as yf

data_file = './research/Generic_Momentum/snp500.pkl'

start_time = datetime(2007, 1, 1)
bm_ticker = '^GSPC'

# get snp 500 tickets
if( not os.path.exists(data_file) ):
    # S&P500 dataframe: list of tickers

    sp_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp_df['Symbol'] = sp_df['Symbol'].str.replace('.', '-')
    tickers_list = [bm_ticker] + list(sp_df['Symbol'])[:]

    # Variables instantiation
    show_batch = True
    df_abs = pd.DataFrame()
    df_abs1 = pd.DataFrame()
    df_abs2 = pd.DataFrame()
    batch_size = 20
    loop_size = int(len(tickers_list) // batch_size) + 2

    cols = ['Close', 'Open', 'Volume']

    for t in range(1,loop_size-1): # Batch download
        m = (t - 1) * batch_size
        n = t * batch_size
        batch_list = tickers_list[m:n]
        print(batch_list,m,n)
        batch_download = yf.download(tickers= batch_list,start= start_time, end = None, 
                            interval = "1d",group_by = 'column',auto_adjust = True, 
                                prepost = True, treads = True, proxy = None)[cols]
        if t == 1:
            df_abs = batch_download
        else:
            s1 = df_abs.stack(level=0)
            s2 = batch_download.stack(level=0)
            s1.loc[:,s2.columns] = s2
            df_abs = s1.unstack().swaplevel(axis=1)

    df_abs = df_abs.loc[:,df_abs.columns.get_level_values(0).unique()]
    df_abs.to_pickle(data_file)

else: 

    df_abs = pd.read_pickle(data_file)



df_abs.fillna(method='ffill', inplace=True)
df_abs.fillna(0.0, inplace=True)
assets = df_abs.columns.get_level_values(1).unique().values.tolist()
assets.remove(bm_ticker)
#assets = df_abs.loc[:, df_abs.columns != bm_ticker].columns.values
print(assets)
print(datetime.now())

class m_PandasData(bt.feeds.PandasData):
   # lines = ('adj_close',)
    params = (
        ('datetime', None),
        ('Date', 'Date'),
        ('open', 'Open'),
        ('high',None),
        ('low',None),
        ('close','Close'),
        ('volume',None),
        ('openinterest',None),
    #    ('adj_close','Adj Close'),
    )

df_abs = df_abs.iloc[:1000]

config_cerebro = {
    'assets':assets[:50],
    'benchmark':bm_ticker,
    'startd': start_time,
    'endd'  : datetime.now(), #datetime(2020, 12, 31),
    'cheat_on' : 'coo',
    'cash': 10000.0,
    'stocks_df':None,
    'benchmark_df':None,
    'generate_report': True,
    'generate_global_report': False,
    'report_name': 'test',
    'warmup_period': 252,
    'data_format': m_PandasData,
    'printResults': True,
    'addBenchmarktToOutput' : True,
    'maxcpus' : 1


}

params1 = { 
    'm_name' : "Generic_Momentum",
    'WarmUpPeriod': config_cerebro['warmup_period'],
    'RebalanceDay': 22,
    'reserveCash': 1000.0,
    'printlog': 0,
    'benchmarkUse': True,

    'selectionModel' : { 
        'model' : SelectionModel_DropNa,
        'params' : {}
    },
    'alphaModel' : { 
        'model' : AlphaModel_GenericMomentum,
        'params' : {
            'n_top': 10,
        }
    },
    'portfolioConstructionModel' : { 
        'model' : AllocationModel_Equal,
        'params' : {}
    },
    'rebalanceModel' : {
        'model' : RebalanceModel_Simple,
        'params' : {}
    },
    'riskModel' : {
        'model' : None,
        'params' : {}
    },
    'executionModel' : {
        'model' : None,
        'params' : {}
    },
}

input_param = {'inputs':[params1]}

cols = df_abs.columns.get_level_values(0).unique().values
config_cerebro['stocks_df'] = df_abs.loc[:, (tuple(cols), assets)]
config_cerebro['benchmark_df'] = df_abs.loc[:, (tuple(cols), [bm_ticker])].droplevel(1,axis=1)

df = backtest(config_cerebro, DynRebalance, **input_param)

#print(df.columns)