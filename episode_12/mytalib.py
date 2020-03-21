import pandas as pd
import numpy as np

#Moving Average  
def MA(df, n, col='close', _name=''):  
    if not _name:
        _name = 'MA_'+ col+ '_' + str(n)
    MA = pd.Series(df[col]).rolling(window=n).mean().fillna(df[col])
    df.loc[:,_name] = MA
    return df

#Exponential Moving Average  
def EMA(df, n, col='close', _name=''):  
    if not _name:
        _name = 'EMA_'+ col+ '_' + str(n)
    EMA = pd.DataFrame.ewm(df[col], span = n, min_periods = n - 1).mean().fillna(df[col])
    df.loc[:,_name] = EMA
    return df    

#RSI
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi    

def RSI(df, n, col='close', _name=''):
    if not _name:
        _name = 'RSI_'+ col+ '_' + str(n)
    df.loc[:,_name] = computeRSI(df[col], n)
    df.loc[:,_name] = df[_name].fillna(df[col])
    return df

