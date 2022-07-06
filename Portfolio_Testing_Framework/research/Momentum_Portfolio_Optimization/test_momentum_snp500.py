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
import quantstats


from base_AllocationModel import Base_AllocationModel
from SelectionModelAll import SelectionModel_All
from SelectionModelDropNa import SelectionModel_DropNa

from AlphaModelRandom import AlphaModel_Random
from AlphaModelMomentum import AlphaModel_Momentum
from AllocationModelEqual import AllocationModel_Equal
from AllocationModelPyFo import AllocationModel_PyFo
from RebalanceModelSimple import RebalanceModel_Simple
from RiskModel_StopLoss import RiskModel_StopLoss

from bt_testcode3 import backtest, DynRebalance

from helper_lib import storeDictionaryToFile

from pandas_datareader import data as pdr
import yfinance as yf

data_file = './research/Momentum_Portfolio_Optimization/snp500_2000.pkl'

start_time = datetime(2000, 1, 1)
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
        ('open','Open'),
        ('high',None),
        ('low',None),
        ('close','Close'),
        ('volume',None),
        ('openinterest',None),
    #    ('adj_close','Adj Close'),
    )

config_cerebro = {
    'assets':assets,
    'benchmark':bm_ticker,
    'startd': datetime(2004, 1, 1),
    'endd'  : datetime(2016, 1, 1), 
    'cheat_on' : 'coo',
    'cash': 10000.0,
    'stocks_df':None,
    'benchmark_df':None,
    'generate_report': True,
    'generate_global_report': True,
    'report_name': 'test',
    'warmup_period': 252,
    'data_format': m_PandasData,
    'printResults': True,
    'addBenchmarktToOutput' : True,
    'maxcpus' : 1
}


################################################
#     Momentum with equal allocation model     #
################################################
params_equal = { 
    'm_name' : "Momentum_Equal",
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
        'model' : AlphaModel_Momentum,
        'params' : {
            'isLongActive' : True,
            'n_top': 10,
            'isShortActive' : False,
            'n_bot': 2,
            'isFIPUsed' : False,
            'n_FIP': 20,
            'momentum_type' : 'generic_momentum',
            'momentum_window' : 252,
            'invertMomentum' : False
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
        'model' : RiskModel_StopLoss,
        'params' : {'stopLoss': 0.00,
                    'resetRebalanceCounter' : True
        }
    },
    'executionModel' : {
        'model' : None,
        'params' : {}
    },
}

################################################
#     Momentum with FIP                        #
################################################
params_e_fip = copy.deepcopy(params_equal)
params_e_fip['m_name'] = "Momentum_FIP"
params_e_fip['alphaModel']['params']['isFIPUsed'] = True
params_e_fip['alphaModel']['params']['n_FIP'] = 20

################################################
#     Momentum with CVaR model                 #
################################################
params_cvar = copy.deepcopy(params_equal)
params_cvar['m_name'] = "Momentum_CVaR"
params_cvar['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_cvar['portfolioConstructionModel']['params']['method'] = 'CVaR'

################################################
#     Momentum with CDaR model                 #
################################################
params_cdar = copy.deepcopy(params_equal)
params_cdar['m_name'] = "Momentum_CDaR"
params_cdar['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_cdar['portfolioConstructionModel']['params']['method'] = 'CDaR'

################################################
#     Momentum with CLA model                  #
################################################
params_cla = copy.deepcopy(params_equal)
params_cla['m_name'] = "Momentum_CLA"
params_cla['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_cla['portfolioConstructionModel']['params']['method'] = 'CLA'

################################################
#     Momentum with HRP model                  #
################################################
params_hrp = copy.deepcopy(params_equal)
params_hrp['m_name'] = "Momentum_HRP"
params_hrp['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_hrp['portfolioConstructionModel']['params']['method'] = 'HRP'

################################################
#    MA Momentum with equal allocation model   #
################################################
params_ma_equal = copy.deepcopy(params_equal)
params_ma_equal['m_name'] = "MA_Momentum_equal"
params_ma_equal['alphaModel']['params']['momentum_type'] = 'ma_momentum'

################################################
#     MA Momentum with equal CVaR model        #
################################################
params_ma_cvar = copy.deepcopy(params_ma_equal)
params_ma_cvar['m_name'] = "MA_Momentum_CVaR"
params_ma_cvar['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ma_cvar['portfolioConstructionModel']['params']['method'] = 'CVaR'

################################################
#     MA Momentum with equal CDaR model        #
################################################
params_ma_cdar = copy.deepcopy(params_ma_equal)
params_ma_cdar['m_name'] = "MA_Momentum_CDaR"
params_ma_cdar['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ma_cdar['portfolioConstructionModel']['params']['method'] = 'CDaR'

################################################
#     Momentum with CLA model                  #
################################################
params_ma_cla = copy.deepcopy(params_ma_equal)
params_ma_cla['m_name'] = "MA_Momentum_CLA"
params_ma_cla['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ma_cla['portfolioConstructionModel']['params']['method'] = 'CLA'

################################################
#     Momentum with HRP model                  #
################################################
params_ma_hrp = copy.deepcopy(params_ma_equal)
params_ma_hrp['m_name'] = "MA_Momentum_HRP"
params_ma_hrp['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ma_hrp['portfolioConstructionModel']['params']['method'] = 'HRP'

################################################
#     Momentum with EF model                   #
################################################
params_ef = copy.deepcopy(params_equal)
params_ef['m_name'] = "Momentum_E_Frontier"
params_ef['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ef['portfolioConstructionModel']['params']['method'] = 'E_Frontier'

################################################
#     MA Momentum with EF model                   #
################################################
params_ma_ef = copy.deepcopy(params_equal)
params_ma_ef['m_name'] = "MA Momentum_E_Frontier"
params_ma_ef['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ma_ef['portfolioConstructionModel']['params']['method'] = 'E_Frontier'

################################################
#     Random Alpha model                       #
################################################
params_random = copy.deepcopy(params_equal)
params_random['m_name'] = "Random"
params_random['alphaModel']['model'] = AlphaModel_Random
params_random['alphaModel']['params']['seed'] = 100

################################################
#     MA Momentum with equal CDaR model SL     #
################################################
params_ma_cdar_SL = copy.deepcopy(params_ma_equal)
params_ma_cdar_SL['m_name'] = "MA_Momentum_CDaR_SL"
params_ma_cdar_SL['portfolioConstructionModel']['model'] = AllocationModel_PyFo
params_ma_cdar_SL['portfolioConstructionModel']['params']['method'] = 'CDaR'
params_ma_cdar_SL['riskModel']['params']['stopLoss'] = 0.15


#input_param = {'inputs':[params_equal, params_ma_equal, params_cla, params_ma_cla, params_hrp, params_ma_hrp, params_cvar, params_ma_cvar, params_cdar, params_ma_cdar]}
input_param = {'inputs':[params_equal, params_ma_equal]}

cols = df_abs.columns.get_level_values(0).unique().values
config_cerebro['stocks_df'] = df_abs.loc[:, (tuple(cols), assets)]
config_cerebro['benchmark_df'] = df_abs.loc[:, (tuple(cols), [bm_ticker])].droplevel(1,axis=1)

df_statistics, allocation_history, df_returns = backtest(config_cerebro, DynRebalance, **input_param)

storeDictionaryToFile('allocation_history.pickle', allocation_history)
df_returns.to_csv('returns_history.csv')
#quantstats.reports.html(df_returns.loc[:, 'S_MA_Momemtum'], benchmark=df_returns.loc[:, 'S_Equal'], output='test1.html', download_filename='test1.html', title="MA_Momentum vs Momentum")
