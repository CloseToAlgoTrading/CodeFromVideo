from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from telnetlib import STATUS
import numpy as np
import pandas as pd
from enum import IntEnum
from tqdm import tqdm

import backtrader as bt
from datetime import datetime
import backtrader.analyzers as btanalyzers
from pandas_datareader import data as pdr

#Following code need to be here for simple access to the modules
import sys
sys.path.insert(0, './../../common')
sys.path.insert(0, './../../Models')
sys.path.insert(0, './common')
sys.path.insert(0, './Models')

from base_riskModel import Base_RiskModel
from base_executionModel import Base_ExecutionModel
from base_SelectionModel import Base_SelectionModel
from base_AlphaModel import Base_AlphaModel
from base_RebalanceModel import Base_RebalanceModel
from base_AllocationModel import Base_AllocationModel

import operator

from backtrader.utils.py3 import map
from backtrader import Analyzer, TimeFrame
from backtrader.analyzers import AnnualReturn

import quantstats



float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

np.random.seed(123)

#------------------- Analysers -----------
# backtrader
class AllocationHistory(Analyzer):
    params = (('timeframe', TimeFrame.Days),)

    def __init__(self):
        super(AllocationHistory, self).__init__()
        self.df_allocation = pd.DataFrame()
       

    def start(self):
        # Not needed ... but could be used
        pass

    def next(self):
        d = [self.strategy.datetime.date(ago=0)]
        d = {'Date': self.strategy.datetime.date(ago=0)}
        d.update(self.strategy.t_allocation)
        self.df_allocation = pd.concat([self.df_allocation, pd.DataFrame([d])])
        pass

    def stop(self):
        self.df_allocation = self.df_allocation.fillna(0.0)
        self.df_allocation.set_index('Date', inplace=True)
        pass

    def get_analysis(self):
        return dict(allocation=self.df_allocation)




def generateOuputResult(s_name, returns):

    mtx = quantstats.reports.metrics(returns=returns,
                rf=0., display=False, mode='full',
                compounded=True,
                periods_per_year=252,
                prepare_returns=False)
    mtx.rename(columns = {'Strategy':s_name}, inplace = True)
    return mtx.T.loc[:,['Cumulative Return', 'CAGR﹪', 'Volatility (ann.)', 'Sharpe','Max Drawdown']]


class DynRebalance(bt.Strategy):
    
    class LogLevel(IntEnum):
        L_1 = 1,
        L_2 = 2,
        L_3 = 3

    def getAllCash(self):

        openPos = self.getOpenPositions()
        cash = self.broker.get_cash()# - self.reserveCash
        pv = self.broker.get_value()# - self.reserveCash

        pnl = 0
        free_cash = 0
        for a, v in openPos.items():
            pnl += v['size']*(v['currentPrice'] - v['openPrice'])
            free_cash += v['size']*v['openPrice']

        all_cash_pnl = cash + free_cash + pnl
        #print('all_cash_pnl', all_cash_pnl)        
        return all_cash_pnl

    def getOpenPositions(self):
        p = {}
        for i in range(self.nDatas):
            s = self.getposition(data=self.datas[i]).size
            if(s != 0):
                p[self.datas[i]._name] = {  'size':s, 
                                            'openPrice':self.getposition(data=self.datas[i]).price,
                                            'currentPrice':self.datas[i].close[0]
                                        }
        return p

    def printAssets(self):
        for i in range(self.nDatas):
            print(i, self.datas[i]._id, self.datas[i]._name)

    def getAssetsDict(self):
        ret = {}
        for i in range(self.nDatas):
            ret[self.datas[i]._name] = self.datas[i]._id
        return ret

    def converToDataFrame(self, data, nDatas, historySteps):
        p = {}
        if(nDatas > 0):
            datat4 = []
            for i in reversed(range(historySteps)):    
                datat4.append(data[0].datetime.datetime(-i))
            for i in range(nDatas):
                p[data[i]._name] = np.array(data[i].get(size=historySteps))
            df = pd.DataFrame(p, index=datat4)

        return df

    def Log(self, verboseLevel, msg):
        if (self.verbose >= verboseLevel):
            print(msg)
        

    def __init__(self, m_input=None):

        self.t_allocation = []
        self.final_value = 0

        #deifne warup parametres        
        self.WarmUpCounter = 0

        self.strategy_name = m_input.get('m_name','NA')

        self.WarmUpPeriod = m_input.get('WarmUpPeriod', 0) #125
        
        self.historyPeriod = m_input.get('WarmUpPeriod', 252) #125

        self.update_counter = 0
        #if the last element is benchmark reduce number of datas
        if(m_input.get('benchmarkUse', False) == True):
            self.nDatas = len(self.datas)-1
        else:
            self.nDatas = len(self.datas)
        

        self.RebalanceDay = m_input.get('RebalanceDay', 1) #22
        self.reserveCash = m_input.get('reserveCash',0) #1000.0

        self.verbose = m_input.get('printlog',0) #0

        self.Assets = {}
        for v,k in enumerate(self.datas):
            if k._name not in self.Assets:
                self.Assets[k._name] = v

        #print(self.Assets)

        if(m_input['selectionModel']['model'] is not None):
            self.selectionModel = m_input['selectionModel']['model'](self, self.getAssetsDict(), **m_input['selectionModel']['params'])
        else:
            self.selectionModel = Base_SelectionModel(self, self.getAssetsDict())

        if(m_input['alphaModel']['model'] is not None):
            self.alphaModel = m_input['alphaModel']['model'](self, **m_input['alphaModel']['params'])
        else:
            self.alphaModel = Base_AlphaModel(self)

        if(m_input['portfolioConstructionModel']['model'] is not None):
            self.portfolioConstructionModel = m_input['portfolioConstructionModel']['model'](self, **m_input['portfolioConstructionModel']['params'])
        else:
            self.portfolioConstructionModel = Base_AllocationModel(self)

        if(m_input['rebalanceModel']['model'] is not None):
            self.rebalanceModel = m_input['rebalanceModel']['model'](self, **m_input['rebalanceModel']['params'])
        else:
            self.rebalanceModel = Base_RebalanceModel(self)

        if(m_input['executionModel']['model'] is not None):
            self.executionModel = m_input['executionModel']['model'](self, **m_input['executionModel']['params'])
        else:
            self.executionModel = Base_ExecutionModel(self)
        
        if(m_input['riskModel']['model'] is not None):
            self.riskModel = m_input['riskModel']['model'](self, **m_input['riskModel']['params'])
        else:
            self.riskModel = Base_RiskModel(self)

        #self.printAssets()

        if self.verbose == 0:
           self.pbar = tqdm(total=len(self.datas[0].array),ascii=' =')
           self.pbar.set_description(self.strategy_name)

        pass

    def next(self):
        if self.verbose == 0:
            self.pbar.update(1)
        # warm up period
        if self.WarmUpCounter < self.WarmUpPeriod:
            self.WarmUpCounter += 1
            return

        allocations = {}

        # check if we need to reset Model
        if self.update_counter == 0:
            #construct data frame
            m_input2 = self.converToDataFrame(self.datas, self.nDatas, self.historyPeriod)

            #select/filter stocks
            selected_stocks = self.selectionModel.get_assets(m_input2)
            self.Log(self.LogLevel.L_1, '----------------')
            #run an alpha model to generate signals
            selected_signals = self.alphaModel.get_signals(selected_stocks, m_input2)
            self.Log(self.LogLevel.L_1, f'selected_signals {selected_signals}')
            #get portfolio allocation 
            allocations = self.portfolioConstructionModel.get_allocations(selected_signals, m_input2)
            self.Log(self.LogLevel.L_1, f'allocations {allocations}')
            #tmp
            self.t_allocation = allocations
            #get rebalance actions
            allocations = self.rebalanceModel.get_allocations(allocations, m_input2)
            self.Log(self.LogLevel.L_1, f'size_allocations {allocations}')


        allocations = self.riskModel.get_allocations(allocations)
        self.Log(self.LogLevel.L_2, f'risk -> size_allocations {allocations}')

        self.executionModel.execute(allocations)
            

        self.update_counter += 1
        if self.update_counter >= self.RebalanceDay:
            self.update_counter = 0

    def stop(self):
        self.final_value = round(self.broker.get_value(), 2)
        if self.verbose == 0:
            self.pbar.close()
        pass

class PandasData(bt.feeds.PandasData):
    lines = ('adj_close',)
    params = (
        ('datetime', None),
        ('Date', 'Date'),
        ('open','Open'),
        ('high','High'),
        ('low','Low'),
        ('close','Close'),
        ('volume','Volume'),
        ('openinterest',None),
        ('adj_close','Adj Close'),
    )

def backtest(cfg, strategy, **kwargs):
    
    print('Backtest started')

    def getStock(df, s):
        idx = pd.IndexSlice
        c = df.loc[:,idx[:,s]]
        c.columns = c.columns.droplevel(1)
        c = pd.DataFrame(c.to_records()).set_index('Date')
        return c

    df = pd.DataFrame()

    if 'inputs' not in kwargs:
        return df
    if len(kwargs['inputs']) <= 0:
        return df

    #inititalize backtrader
    cerebro = bt.Cerebro(optreturn=False)
   
    #configuration of cheat-on-
    if('coc' == cfg.get('cheat_on', 'coc')):
        #Configure the Cheat-On-Close method to buy the close on order bar
        cerebro.broker.set_coc(True)
        cerebro.broker.set_coo(False)
    else:
        #Configure the Cheat-On-Open method to buy the close on order bar
        cerebro.broker.set_coc(False)
        cerebro.broker.set_coo(True)
        

    inputs = [v for v in kwargs['inputs']]
    # add strategy to the backtrader
    #cerebro.addstrategy(strategy)
    cerebro.optstrategy(strategy, m_input = inputs)

    # adding multiply assets 
    for a in cfg['assets']:
        # if stock data frame is None, we will get data from the yahoo finance
        if cfg.get('stocks_df', None) is None:
            cerebro.adddata(bt.feeds.YahooFinanceData(dataname=a, fromdate=cfg['startd'], todate=cfg['endd'], plot=False))
        else:
            f = (cfg['stocks_df'].index >= cfg['startd']) & (cfg['stocks_df'].index < cfg['endd'])
            cfg['stocks_df'] = cfg['stocks_df'].loc[f]
            # else from the pandas dataframe
            if cfg.get('data_format', None) is not None:
                cerebro.adddata(cfg['data_format'](dataname=getStock(cfg['stocks_df'],a)), name=a)
            else:
                cerebro.adddata(PandasData(dataname=getStock(cfg['stocks_df'],a)), name=a)
        
    #add benchmark 
    if 'benchmark' in cfg:
        if (cfg['benchmark_df'] is None):
            bm = bt.feeds.YahooFinanceData(dataname=cfg['benchmark'], fromdate=cfg['startd'], todate=cfg['endd'], plot=False)
        else:
            f = (cfg['benchmark_df'].index >= cfg['startd']) & (cfg['benchmark_df'].index < cfg['endd'])
            cfg['benchmark_df'] = cfg['benchmark_df'].loc[f]
            if('data_format' in cfg):
                bm = cfg['data_format'](dataname=cfg['benchmark_df'],name=cfg['benchmark'])
            else:
                bm = PandasData(dataname=cfg['benchmark_df'],name=cfg['benchmark'])

        cerebro.adddata(bm)
    
    if(True == cfg.get('addBenchmarktToOutput', False)):
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, data=bm, _name='BenchmarkReturns')
    cerebro.addanalyzer(AllocationHistory, _name='allocationHistory')

    # add pyFolio Analyzer
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')

    
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.setcash(cfg['cash'])
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())


    strats = cerebro.run(maxcpus = cfg.get('maxcpus',1))

    allocation_history = {}
    bm_ret = None
    df_returns = pd.DataFrame()
    print('Report:')
    for run in tqdm(strats,ascii=' ='):
        for strategy in  run:
            bm = cfg.get('benchmark', None)
            if (bm_ret is None) and (bm is not None) and (True == cfg.get('addBenchmarktToOutput', False)):
                bm_ret = strategy.analyzers.getbyname('BenchmarkReturns').get_analysis()
                lists = sorted(bm_ret.items())
                x, y = zip(*lists)
                bm_ret = pd.DataFrame({'data':x, 'Benchmark':y}).set_index('data')
                bm_ret = bm_ret.iloc[strategy.WarmUpPeriod:]
                ret = generateOuputResult(bm, bm_ret.loc[:, 'Benchmark'])
                df = pd.concat([df, ret])


            portfolio_stats = strategy.analyzers.getbyname('PyFolio')
            returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
            if transactions.size > 0:
                returns.name = strategy.strategy_name
                returns.index = returns.index.tz_convert(None)
                returns = returns.iloc[strategy.WarmUpPeriod:]
                df_returns = pd.concat([df_returns, returns.copy()],axis=1)
                if True == cfg.get('generate_report', False):
                    fname = (cfg.get('report_name', 'test_report')).replace(" ", "_") + '_' + strategy.strategy_name + '.html'
                    bm = cfg.get('benchmark', None)
                    if (bm_ret is not None):
                        quantstats.reports.html(returns, benchmark=bm_ret, output=fname, download_filename=fname, title=cfg.get('report_name', 'test_report'))
                    else:
                        quantstats.reports.html(returns, output=fname, download_filename=fname, title=cfg.get('report_name', 'test_report'))

                ret = generateOuputResult(strategy.strategy_name, returns)
                df = pd.concat([df, ret])

                allocation_history[strategy.strategy_name] = strategy.analyzers.getbyname('allocationHistory').get_analysis()['allocation'].iloc[strategy.WarmUpPeriod:]


    df_returns.index = pd.to_datetime(df_returns.index)
    if True == cfg.get('generate_global_report', False) and (df_returns.shape[0] > 0):
        fname = (cfg.get('report_name', 'test_report')).replace(" ", "_") + '_GLOBAL_' + '.html'
        if (bm_ret is not None):
            quantstats.reports.html2(df_returns, benchmark=bm_ret, output=fname, download_filename=fname, title=cfg.get('report_name', 'test_report'), template_path='./reports_temlate/report.html')
        else:
            quantstats.reports.html2(df_returns, output=fname, download_filename=fname, title=cfg.get('report_name', 'test_report'), template_path='./reports_temlate/report.html')


    if(True == cfg.get('printResults',False)):
        print('\nResults:')
        print(df)
    if (bm_ret is not None):
        df_returns = pd.concat([df_returns, bm_ret],axis=1)
    return df, allocation_history, df_returns
