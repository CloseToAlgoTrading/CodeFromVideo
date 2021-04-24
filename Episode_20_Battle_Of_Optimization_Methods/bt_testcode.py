from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import pandas as pd

import backtrader as bt
from datetime import datetime
import backtrader.analyzers as btanalyzers
import plotly.graph_objects as go
from pandas_datareader import data as pdr


from absl import logging


import operator

from backtrader.utils.py3 import map
from backtrader import Analyzer, TimeFrame
from backtrader.analyzers import AnnualReturn

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

np.random.seed(123)
very_small_float = 1.0 #0.000000001

# With this function we can extract data from the multiindex data frame
# input df - dataframe, s - ticker
idx = pd.IndexSlice
def getStock(df, s):
    c = df.loc[:,idx[:,s]]
    c.columns = c.columns.droplevel(1)
    c = pd.DataFrame(c.to_records()).set_index('Date')
    c = c.fillna(very_small_float)
    return c

# Plot assset allocation history
# input is a dataframe with allocations
def plotAllocation(alloc):
    fig = go.Figure()
    for c in alloc.columns:
        fig.add_trace(go.Bar(x=alloc.index.values, y=alloc[c], name=c, ))
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'}, bargap=0.0,bargroupgap=0.0)
    fig.show()

#delete raws with 0.0 from the dataframe
def cleanWarmUperiod(df):
    df2 = df.copy()
    f = df2.Portfolio == 0
    df2.loc[f] = 0
    return df2

#delete raws with 0.0 from the dataframe
def cleanWarmUperiod(df):
    df2 = df.copy()
    f = df2.Portfolio == 0
    df2.loc[f] = 0
    return df2

#calculates potfolio statistics based on allocation
#input alloc - allocation array, stock_close - dataframe with close prices
def calc_portfolio_values(alloc, stock_close):
    pct = np.divide(stock_close.values, stock_close.values[0])
    portfolio_values = np.sum(pct * alloc, axis=1)
    portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns)
    print('portfolio_returns: {:.4f}'.format( np.sum(portfolio_returns )))
    print('std: {:.4f}'.format(np.std(portfolio_returns)))
    print('sharpe: {:.4f}'.format( sharpe ))

def get_portfolio_values(alloc, stock_close):
    pct = np.divide(stock_close.values, stock_close.values[0])
    portfolio_values = np.sum(pct * alloc, axis=1)
    portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns)
    return (np.sum(portfolio_returns ), np.std(portfolio_returns), sharpe)

#get name from the parameter dictionary
def get_name(params):
    dc = ''
    if params['resetModel'] == True:
        dc += '_rm'
    if params['model_params']['collectData'] == True:
        dc += '_cd'
    s = params['model_params']['model_n'] + dc
    return s



#create dataframe from the dictionary with statistics
def getStatistic(d_stat):
    df = pd.DataFrame()
    for v in d_stat.values():
        df = df.append(v)
    return df

#plot portfolio return from the dictionary with returns
def plotPortfolioReturns(d_res):
    df = pd.DataFrame()
    for i,(k,v) in enumerate(d_res.items()):
        if i == 0:
            df['Benchmark'] = v.Benchmark
        df[k] = v.Portfolio

    df.plot(figsize=(10,8))
    return df

def plotPorfolioReturns(d_res, fn = None):
    df = pd.DataFrame()
    for i,(k,v) in enumerate(d_res.items()):
        if i == 0:
            df['Benchmark'] = v.Benchmark
        df[k] = v.Portfolio

    if fn is not None:
        lstm_dd = pd.read_csv(fn, index_col=0)
        for c in lstm_dd.columns:
            if c != 'Benchmark':
                df[c] = lstm_dd[c]

    df.plot(figsize=(10,8))
    return df


# backtrader
class AllocationHistory(Analyzer):
    params = (('timeframe', TimeFrame.Days),)

    def __init__(self):
        super(AllocationHistory, self).__init__()
        self.alloc = []
        self.name_dic = ['date']

        for num in range(len(self.datas)-1):
            self.name_dic.append(self.datas[num]._name)
        
       # print(self.name_dic)
        

    def start(self):
        # Not needed ... but could be used
        pass

    def next(self):
        # Not needed ... but could be used
        #print(self.strategy.datetime.date(ago=0))
        d = [self.strategy.datetime.date(ago=0)]
        d.extend(self.strategy.new_pct)
        self.alloc.append(d)
        pass

    def stop(self):
        pass

    def get_analysis(self):
        #print(self.alloc)
        #print(np.array(self.name_dic).shape)
        df = pd.DataFrame(self.alloc, columns=self.name_dic).set_index('date')
        return dict(allocation=df)

class DynRebalance(bt.SignalStrategy):
    
    params = (
    ('DataCounter', 125),
    ('RebalanceDay', 22),
    ('reserveCash', 1000.0),
    ('printlog', 0),
    ('benchmarkUse', True),
    ('resetModel',False), 
    ('resetModelStep',4),
    ('model',None),
    ('model_params',{})
    )
    
    def getPosDiffiretce(self, cash, alloc, new_price, cur_pos):
        pos_cash = new_price * cur_pos
        #print('pos_cash',pos_cash)
        all_cash = cash + np.sum(pos_cash)
        #print('all_cash',all_cash)
        cash_alloc = alloc * all_cash
        #print('cash_alloc',cash_alloc)
        new_pos = (cash_alloc / new_price).astype(int)
        #print('new_pos',new_pos)
        diff_pos = cur_pos - new_pos
        return diff_pos*(-1)

    def getPosSize(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.getposition(data=self.datas[i]).size)
        return np.array(p)
    
    def getCurrentClosePrice(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.datas[i].close[0])
        return np.array(p)

    def getPosOpenPrice(self):
        p = []
        for i in range(self.nDatas):
            p.append(self.getposition(data=self.datas[i]).price)
        return np.array(p)
    
    def getModelDataFrame(self):
        p = {}
        for i in range(self.nDatas):
            p[str(i)] = np.array(self.datas[i].get(size=self.DataCounter))
        return pd.DataFrame(p)
    
    def __init__(self):
        self.counter = 0
        self.update_counter = 0
        self.model = self.params.model
        self.isFirst = True
        self.old_pct = []
        self.new_pct = []
        #if the last element is benchmark reduce number of datas
        if(self.params.benchmarkUse == True):
            self.nDatas = len(self.datas)-1
        else:
            self.nDatas = len(self.datas)
        self.DataCounter = self.params.DataCounter #125
        self.RebalanceDay = self.params.RebalanceDay #22
        self.reserveCash = self.params.reserveCash #1000.0

        self.verbose = self.params.printlog #0
        self.resetModel = self.params.resetModel
        self.model_n = self.params.model_params['model_n']
        self.resetModelStep = self.params.resetModelStep
        self.model_params = self.params.model_params
        self.resetModelCounter = 0 

        pass

    def next(self):
        if self.counter < self.DataCounter:
            self.counter += 1
            return

        m_input = self.getModelDataFrame()
        if self.update_counter == 0:
            if self.resetModel == True:
                self.resetModelCounter += 1
                if(self.resetModelCounter == self.resetModelStep):
             #       print('ResetModel')
                    self.model.resetModel()
                    self.resetModelCounter = 0
  
            self.new_pct = np.round(self.model.get_allocations(m_input.values, **self.model_params),2)
            
            if self.isFirst==True:
                self.isFirst = False
                self.old_pct = np.zeros(len(self.new_pct))

            if np.array_equal(self.new_pct, self.old_pct) == False:
                if self.verbose > 1:
                    print('size', self.getPosSize().tolist())
                    print('price', self.getPosOpenPrice().tolist())
                if self.verbose > 0:
                    print("rebalance new percent.",self.new_pct)
                
                cash = self.broker.get_cash() - self.reserveCash
                if(cash > self.reserveCash):
                    cash -= self.reserveCash
                    
                new_price = self.getCurrentClosePrice()
                cur_pos = self.getPosSize()
                upd_pos = self.getPosDiffiretce(cash, self.new_pct, new_price, cur_pos)
                to_sell = []
                to_buy  = []
                for i,p in enumerate(upd_pos):
                    if(p<0):
                        to_sell.append((i,p))
                    elif(p>0):
                        to_buy.append((i,p))
                
                for i,p in to_sell:
                    if self.verbose > 0:
                        print('sell',i,p)
                    self.sell(self.datas[i], p)
                for i,p in to_buy:
                    if self.verbose > 0:
                        print('buy',i,p)
                    self.buy(self.datas[i], p)
                    
                self.old_pct = self.new_pct


        self.update_counter += 1
        if self.update_counter == self.RebalanceDay:
            self.update_counter = 0

class PandasData(bt.feeds.PandasData):
    lines = ('adj_close',)
    params = (
        ('datetime', None),
        ('open','Open'),
        ('high','High'),
        ('low','Low'),
        ('close','Close'),
        ('volume','Volume'),
        ('openinterest',None),
        ('adj_close','Adj Close'),
    )

            
def backtest(cfg, strategy, plot=False, m_name='Test', **kwargs):
    
    def getReturnAsDataFrame(pfa, bencha):
        d = bencha.get_analysis()
        t = pfa.get_analysis()
        lists = sorted(d.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        listst = sorted(t.items()) # sorted by key, return a list of tuples
        x, y1 = zip(*listst) # unpack a list of pairs into two tuples
        dd = {'data':x, 'Benchmark':np.cumsum(y), 'Portfolio':np.cumsum(y1)}
        df = pd.DataFrame(dd).set_index('data')
        return df

    cerebro = bt.Cerebro()
    
    if 'set_coc' in cfg:
        cerebro.broker.set_coc(cfg['set_coc'])
    if 'set_coo' in cfg:
        cerebro.broker.set_coo(cfg['set_coo'])
        
    cerebro.addstrategy(strategy, **kwargs)

    for a in cfg['assets']:
        if cfg['stocks_df'] is None:
            cerebro.adddata(bt.feeds.YahooFinanceData(dataname=a, fromdate=cfg['startd'], todate=cfg['endd'], plot=False))
        else:
            cerebro.adddata(PandasData(dataname=getStock(cfg['stocks_df'],a)), name=a)

        
        
    #add benchmark 
    if 'benchmark' in cfg:
        if (cfg['benchmark_df'] is None) and (cfg['stocks_df'] is None):
            bm = bt.feeds.YahooFinanceData(dataname=cfg['benchmark'], fromdate=cfg['startd'], todate=cfg['endd'])
        else:
            bm = PandasData(dataname=cfg['benchmark_df'],name=a)

        cerebro.adddata(bm)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, data=bm, _name='BenchmarkReturns')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='PortfolioReturns')

    # Analyzer
    cerebro.addanalyzer(btanalyzers.SharpeRatio, riskfreerate=0.0, timeframe=bt.TimeFrame.Months, _name='mysharpe')
    cerebro.addanalyzer(btanalyzers.Returns, _name='myreturn')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='mydrawdown')
    cerebro.addanalyzer(AllocationHistory, _name='allocationHistory')
    
    cerebro.broker.setcash(cfg['cash'])
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    thestrats = cerebro.run()
    thestrat = thestrats[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    if plot:
        cerebro.plot()
    
    df = pd.DataFrame()
    if 'benchmark' in cfg:
        df = getReturnAsDataFrame(thestrat.analyzers.getbyname('PortfolioReturns'), 
                                  thestrat.analyzers.getbyname('BenchmarkReturns'))
    
    ret = {
        m_name : { 
        'Max_Drawdown':thestrat.analyzers.getbyname('mydrawdown').get_analysis()['max']['drawdown'],
        'CAGR':thestrat.analyzers.getbyname('myreturn').get_analysis()['rnorm100'],
        'Sharp_Ratio':thestrat.analyzers.getbyname('mysharpe').get_analysis()['sharperatio'],
        'Value': cerebro.broker.getvalue(),
        }
    }

    allocation_history = thestrat.analyzers.getbyname('allocationHistory').get_analysis()['allocation']
    return (pd.DataFrame.from_dict(ret).T,
            cleanWarmUperiod(df), 
            allocation_history)
