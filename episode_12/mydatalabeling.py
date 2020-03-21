import pandas as pd
import numpy as np
from numba import jit
from tqdm import tqdm
import numba  
import random

from scipy.stats import shapiro
from scipy.stats import normaltest

import multiprocessing as mp

disable_tdqm = True

def read_ticks(fp):
    cols = list(map(str.lower,['Date','Time','Price','Bid','Ask','Size']))
    df = (pd.read_csv(fp, header=None)
          .rename(columns=dict(zip(range(len(cols)),cols)))
#          .assign(dates=lambda df: (pd.to_datetime(df['date']+df['time'],
#                                                  format='%m/%d/%Y%H:%M:%S')))
#          .assign(v=lambda df: df['size']) # volume
#          .assign(dv=lambda df: df['price']*df['size']) # dollar volume
#          .drop(['date','time'],axis=1)
          .set_index('date')
          .drop_duplicates())
    df.index = pd.to_datetime(df.index)
    return df
    
    
def volume_bars(df, volume_column, m):
    '''
    compute volume bars

    # args
        df: pd.DataFrame()
        volume_column: name for volume data
        m: int(), threshold value for volume
    # returns
        idx: list of indices
    '''
    #print(m)
    t = df[volume_column]
    max_index = len(df)
    ts = 0
    idx = []
    idx.append(0)
    for i, x in enumerate(tqdm(t, disable = disable_tdqm)):
        ts += x
        if ts >= m:
            #print(ts)
            if(i+1) < max_index:
                idx.append(i+1)
            ts = 0
            continue
    return idx

def volume_bar_df(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx].drop_duplicates()

def volume_bar_start_index(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx].drop_duplicates().index


def get_ohlc(ref, sub_index, _pc='price', _vc='v', ret_orders_df = False, _ac='ask', _bc='bid'):
    '''
    fn: get ohlc from custom bars
    
    # args
        ref : reference pandas series with all prices
        sub : custom tick pandas series
    # returns
        tick_df : dataframe with ohlc values
    '''
    cols2 = ['Nr','time', 'ask','bid','price','v']
    ohlc = []
    if ret_orders_df is True: 
        pdabv = pd.DataFrame(data = None, columns=cols2)
        #pdabv = []
        #display(pdabv)
    #for i in tqdm(range(len(sub_index)-1)):
    for i in tqdm(range(len(sub_index)-1), disable = disable_tdqm):
        start,end = sub_index[i], sub_index[i+1]-1
        #if i < 20:
        #    print(start,end)
        tmp_ref = ref[_pc].loc[start:end]
        max_px, min_px = tmp_ref.max(), tmp_ref.min()
        
        tmp_ref_v = ref[_vc].loc[start:end]
        sum_v = tmp_ref_v.sum()
        #print(start,end-1, sum_v)
        o,h,l,c,v = ref[_pc].iloc[start], max_px, min_px, ref[_pc].iloc[end], sum_v
        
        #if i < 20:
        #    print(ref[['price','v']].iloc[start])
        ohlc.append((start,o,h,l,c,v))
        
        if ret_orders_df is True:
            tmp_abv = ref[['time', _ac, _bc, _pc, _vc]].loc[start:end]
            tmp_abv['Nr'] = start
            pdabv = pdabv.append(tmp_abv[cols2])
            #print(pdabv)
        #start +=1
        
    cols = ['Nr','open','high','low','close', 'sum_vol']
    

    return pd.DataFrame(ohlc,columns=cols).set_index('Nr'), pdabv.set_index('Nr').copy()


def pad(array, reference, offset):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]
    
    # Insert the array in the result at the specified offsets
    result[tuple(insertHere)] = array
    
#   print(result)
    return result


def f1(x):
    ret = np.std(x)*np.sqrt(len(x))
    #print(ret)
    return ret


#function calculates volatility over n random windows 
def get_window_vlt(_df, n_windows, divider, col='pct'):
    '''
    _df - dataframe
    n_windows - number of rumdom windows
    divider - used for reduce max lenth of window
    
    return:
        ls_wsize - list of window size
        vlt_w - volatility for each window
    '''
    #vlt_w = []
    vlt_d = {}
    ls_wsize = random.sample(range(1, int(len(_df)/divider)), n_windows)
    
    #_ret_df = pd.DataFrame(columns=ls_wsize, index=_df.index)
    
    for w in ls_wsize:
        _rt = _df[col].rolling(w).apply(f1, raw=True).median()
        #vlt_w.append(_rt)
        vlt_d[str(w)] = [_rt]
        
    #print(vlt_w)
    _ret_df = pd.DataFrame.from_dict(vlt_d)
    #return ls_wsize, vlt_w, _ret_df
    return _ret_df


def generate_all_vloume_bars(_df, bar_size, volume_M):
    

    _grp = _df.groupby(_df.index)
    _res_df = pd.DataFrame()
    _res_pdf = pd.DataFrame()
    for n,g in _grp:

        g = g.reset_index()
        g.index.name = 'Nr'
        #сalculate volume start index bars for one group
        v_idx_tmp = volume_bar_start_index(g, 'v', volume_M)
        #generate ohlc bars
        v_bars_ohlc_tmp, pdf_tmp = get_ohlc(g, v_idx_tmp, ret_orders_df=True)

        res_m_df = pd.merge(v_bars_ohlc_tmp, g[['date','time']], on='Nr', how='left')
        res_m_df = res_m_df.reset_index().set_index('date')
        
        res_m_pdf = pd.merge(pdf_tmp, g[['date','time']], on='Nr', how='left')
        res_m_pdf = res_m_pdf.reset_index().set_index('date')
        

        _res_df = _res_df.append(res_m_df, sort=False)
        _res_pdf = _res_pdf.append(res_m_pdf, sort=False)
        
    return _res_df, _res_pdf


def generate_all_vloume_bars_executer(g, bar_size):
    
    g = g.reset_index()
    g.index.name = 'Nr'
    #сalculate volume start index bars for one group
    v_idx_tmp = volume_bar_start_index(g, 'v', bar_size)
    #generate ohlc bars
    v_bars_ohlc_tmp, pdf_tmp = get_ohlc(g, v_idx_tmp, ret_orders_df=True)

    res_m_df = pd.merge(v_bars_ohlc_tmp, g[['date','time']], on='Nr', how='left')
    res_m_df = res_m_df.reset_index().set_index('date')

    res_m_pdf = pd.merge(pdf_tmp, g[['date']], on='Nr', how='left')
    res_m_pdf = res_m_pdf.reset_index().set_index('date')

    return (res_m_df, res_m_pdf)

def generate_all_vloume_bars_mp(_df, bar_size):
    

    _grp = _df.groupby(_df.index)
        
    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
        results = pool.starmap(generate_all_vloume_bars_executer, [(g, bar_size) for n,g in _grp])
        
    
    #return results
    results_df = pd.DataFrame()
    results_pdf_df = pd.DataFrame()
    
    for r in results:
        results_df = results_df.append(r[0], sort=False)
        results_pdf_df = results_pdf_df.append(r[1], sort=False)

    return results_df, results_pdf_df


def calculate_mean_win_vlt(_df, n_windows, col='pct'):
    
    _grp = _df.groupby(_df.index)
    _res_df = pd.DataFrame()
    
    vlt_d = {}
    ls_wsize = np.sort(random.sample(range(1, _grp.size().min() ), n_windows))
    
    for n,g in _grp:
        g['pct'] = g.close.pct_change().fillna(0.0).rename('pct')

        for w in ls_wsize:
            #print('w:',w)
            vlt_d[str(w)] = [g[col].rolling(w).apply(f1, raw=True).median()]
            
        _ret_vlt = pd.DataFrame.from_dict(vlt_d)

        _ret_vlt['date'] = g.index[0]
        _ret_vlt = _ret_vlt.set_index('date')
        #print(g.index[0])
        
        _res_df = _res_df.append(_ret_vlt, sort=False)
        
        _res_df = _res_df.fillna(0.0)
        
    return _res_df


def calculate_mean_win_vlt_executer(g, n_windows, ls_wsize, col='pct'):
    
    vlt_d = {}
    g[col] = g.close.pct_change().fillna(0.0).rename(col)

    for w in ls_wsize:
        #print('w:',w)
        vlt_d[str(w)] = [g[col].rolling(w).apply(f1, raw=True).median()]

    _ret_vlt = pd.DataFrame.from_dict(vlt_d)

    _ret_vlt['date'] = g.index[0]
    _ret_vlt = _ret_vlt.set_index('date')
        
    return _ret_vlt

def calculate_mean_win_vlt_mp(_df, n_windows, col='pct'):
    
    _grp = _df.groupby(_df.index)
    
    if _grp.size().min() < n_windows:
        ls_wsize = np.sort(random.sample(range(1, 200), n_windows))
    else:
        ls_wsize = np.sort(random.sample(range(1, _grp.size().min() ), n_windows))
    
    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
        results = pool.starmap(calculate_mean_win_vlt_executer, 
                               [(g, n_windows, ls_wsize, col) for n,g in _grp])
        
    results_df = pd.concat(results)
    results_df = results_df.fillna(0.0)
    
    return results_df


def func_2(p_df, ref_df, window_size, hb, lb, col_ref = 'price', col_close = 'close', verbose=0):

    """
    p_df - from this df the range of indexes will be take
    ref_df - dataframe with the tick data
    windows_size - number of max bar waiting
    hb - high barrier 
    lb - low barrier
    col_ref - name of the price column
    
    return: 1 - touch high barrier first
            0 - no barrier touch
           -1 - touch low barrier first
    """

    min_valid_index = 1
    cross_lst = []

    def getdir(x):
        ret = 0
        
        #get slice from the reference dataframe
        #It is a horizotal window
        loc_df = ref_df.iloc[x.index[0]: x.index[-1]+1]
        
        if(verbose > 1):
            print(loc_df.index)
            print(loc_df)
        
        _x = loc_df[col_ref]
        
        #highest price
        hbl = _x.iloc[0] + hb*_x.iloc[0]
        #lowest price
        lbl = _x.iloc[0] - lb*_x.iloc[0]
        if(verbose > 1):
            print("---->", hbl,lbl)
        
        #do we hit max price?
        fm = _x > hbl 
        #do we hit min price?
        fl = _x < lbl
        
        fmt = (_x > hbl) == True
        flt = (_x < lbl) == True
        
        min_cross = False
        max_cross = False
        
        if (len(fm[fmt])>0):
            max_cross = True
            if(verbose > 1):
                print('val max', fm[fmt].index.values[0])
        if (len(fl[flt])>0):
            min_cross = True
            if(verbose > 1):
                print('val min', fl[flt].index.values[0])


        if max_cross == True and min_cross == True:
            if(fm[fmt].index.values[0] < fl[flt].index.values[0]):
                ret = 1
            else:
                ret = -1
        elif max_cross == True:
            ret = 1
        elif min_cross == True:
            ret = -1
        else:
            ret = 0
        if(verbose > 1):
            print('ret = ', ret)
        
        
        if ret == 1:
            cross_lst.append(fm[fmt].index.values[0])
        if ret == -1:
            cross_lst.append(fl[flt].index.values[0])
        elif ret == 0:
            cross_lst.append(0)
        
        
        return ret

    
    _a = p_df[col_close][::-1].rolling(window_size, min_periods=2).apply(lambda x: getdir(x[::-1]), raw=False)[min_valid_index:].astype(int)
    
    if(verbose > 0):
        print(_a)
        print(cross_lst)
    p_df['dir'] = pd.Series(pad(_a[::-1], p_df[p_df.columns[0]], [min_valid_index])).shift(-min_valid_index).to_list()
    p_df['cross_idx'] = pd.Series(pad(np.array(cross_lst[::-1]), p_df[p_df.columns[0]], [min_valid_index])).shift(-min_valid_index).to_list()

    p_df[['dir','cross_idx']] = p_df[['dir','cross_idx']].fillna(0.0)

    return p_df


def leave_only_one_event(aa, f):
    ld = aa[f].cross_idx - aa[f].index.values
    im = ld.idxmin()
    dir_im = aa.loc[im].dir
    aa.loc[f,'dir'] = 0.0
    aa.loc[im, 'dir'] = dir_im
    return im

def remove_cross_index(aa, start_Index):
    
    f_idx = start_Index
    f = (aa.index >= f_idx) & (aa.dir != 0.0)


    if(len(aa[f]) > 0.0):
        f_idx = aa[f].index[0]
        c_idx = aa[f].loc[f_idx].cross_idx

        f = (aa.dir != 0.0) & (aa.cross_idx <= c_idx) & (aa.index <= c_idx) & (aa.index >= f_idx)
        im = leave_only_one_event(aa, f)
        f2 = (aa.dir != 0.0) & (aa.cross_idx <= c_idx) & (aa.index <= c_idx) & (aa.index >= f_idx)
        f_idx = f_idx+1 
        remove_cross_index(aa, f_idx)
    else:
        f = aa.dir != 0


#generate labels without multiprocessing
def generate_labels(p_df, ref_df, window_size, hb, lb, col_ref = 'price', col_close = 'close', verbose=0, 
                    start_time='10:00:00'):
    
    grp_1 = p_df.groupby(p_df.index)
    grp_2 = ref_df.groupby(ref_df.index)
    
    r_df = pd.DataFrame()
    for g1,gr in zip(grp_1, grp_2):

        g1 =  g1[1].reset_index().set_index('Nr')
        gr =  gr[1].reset_index()
        gr.index.name = 'Nr'
        
        ret_df = func_2(g1 , gr, window_size, hb, lb, col_ref, col_close, verbose)
        
        f = ret_df.time <= start_time

        ret_df.loc[f,'dir'] = 0.0

        remove_cross_index(ret_df, 0)

        r_df = r_df.append(ret_df.copy())
        
    return r_df


def generate_labels_executer(p_df_zip, window_size, hb, lb, col_ref = 'price', col_close = 'close', verbose=0, 
                    start_time='10:00:00'):
    
    g1 = p_df_zip[0]
    gr = p_df_zip[1]
    
    g1 =  g1[1].reset_index().set_index('Nr')
    gr =  gr[1].reset_index()
    gr.index.name = 'Nr'

    ret_df = func_2(g1 , gr, window_size, hb, lb, col_ref, col_close, verbose)

    f = ret_df.time <= start_time

    ret_df.loc[f,'dir'] = 0.0

    remove_cross_index(ret_df, 0)

    return ret_df

#generate labels with multiprocessing
def generate_labels_mp(p_df, ref_df, window_size, hb, lb, col_ref = 'price', col_close = 'close', verbose=0, 
                    start_time='10:00:00'):
    
    grp_1 = list(p_df.groupby(p_df.index))
    grp_2 = list(ref_df.groupby(ref_df.index))
    grp_zip = [x for x in zip(grp_1, grp_2)]
    
    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
        results = pool.starmap(generate_labels_executer, 
                               [(g, 50, 0.003, 0.003, 'price', 'close', 0, '10:00:00') for g in grp_zip])
        
    results_df = pd.concat(results)
    results_df = results_df.reset_index().set_index('date')
    
    return results_df



def all_value_to_cusum_executer(g, col='price'):
    
    g.loc[:, col] = g['price'].diff().cumsum().fillna(0.0000)
    return g

def all_value_to_cusum__mp(_df, col='price'):

    _grp = _df.groupby(_df.index)
        
    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
        results = pool.starmap(all_value_to_cusum_executer, [(g, col) for n,g in _grp])
        
    
    #return results
    results_df = pd.DataFrame()
    
    for r in results:
        results_df = results_df.append(r, sort=False)

    return results_df

    



def metaLabling(close, startEvents, side, verticalBarrier = None, SL = None, TP = None, isPercentUse=False):

    def getVerticalBarriers(_idxs, _wsize=2):
        idxs_len = len(_idxs)
        r = _idxs[-1]
        rl = []
        for i, n in enumerate(_idxs):
            if i+_wsize < idxs_len:
                rl.append(_idxs[i+_wsize-1])
            else:
                rl.append(r)

        return rl


    if verticalBarrier is None:
        _verticalBarrier = pd.Series(close.index[-1], index=startEvents)
    else:
        _verticalBarrier = pd.Series(getVerticalBarriers(close.index.values, verticalBarrier), index=startEvents)

    if TP is None:
        TP = 10.0

    if SL is None:
        SL = 10.0

    if isPercentUse == True:
        ret = pd.DataFrame(columns=['touch_tp', 'raw_return', 'p_return', 'side', 'cross_idx'], index=startEvents)
    else:
        ret = pd.DataFrame(columns=['touch_tp', 'raw_return', 'side', 'cross_idx'], index=startEvents)
    t_touch = 0
    r_ret = 0
    p_ret = 0
    print('startEvents',startEvents)
    for t0 in tqdm(startEvents):
   # for t0 in startEvents:
        window=close.loc[t0:_verticalBarrier[t0]]
        if isPercentUse == True:
            if(side[t0] == 1):
                trgHi = close[t0] * TP
                trgLow = close[t0] * SL
            else:
                trgHi = close[t0] * SL
                trgLow = close[t0] * TP
        else:
            if(side[t0] == 1):
                trgHi = TP #close[t0] + TP
                trgLow = SL #close[t0] - SL
            else:
                trgHi = SL #close[t0] - SL
                trgLow = TP #close[t0] + TP

        #print('\n_verticalBarrier[t0]',_verticalBarrier[t0])
        #print('trgHi', trgHi)
        #print('trgLow', trgLow)
        
        #print('close[t0]', close[t0])
        #print('(trgHi+close[t0])', (trgHi+close[t0]))
        #print('(close[t0]-trgLow)', (close[t0]-trgLow))
        fabove = window >= (trgHi+close[t0])
        fbelow = window <= (close[t0]-trgLow)
        
        hiTouch = window[fabove].index.min()
        lowTouch = window[fbelow].index.min()

        #print('t0', t0)
        #print('ht',(hiTouch))
        #print('lt',(lowTouch))
        _touchidx = 0
        
        if ((pd.isnull(hiTouch)) and (pd.isnull(lowTouch))):
            #display('vertical barrier or infinit')
            t_touch = 0
            r_ret = close.loc[_verticalBarrier[t0]]
            #print('v_barrier ->', t0, _verticalBarrier[t0], close.loc[_verticalBarrier[t0]])
            _touchidx = _verticalBarrier[t0]
            
        elif (pd.isnull(hiTouch) and (not pd.isnull(lowTouch))):
            #display('low touch')
            t_touch = -1 
            r_ret = close.loc[lowTouch]
            _touchidx = lowTouch
            #print('low touch ->', t0, lowTouch, close.loc[lowTouch])
        elif (not pd.isnull(hiTouch) and (pd.isnull(lowTouch))):
            #display('hi touch')
            t_touch = 1 
            r_ret = close.loc[hiTouch]
            _touchidx = hiTouch
            #print('hi touch ->', t0, hiTouch, close.loc[hiTouch])
        elif(hiTouch < lowTouch):
            #display('HI')
            t_touch = 1
            r_ret = close[hiTouch]
            _touchidx = hiTouch
            #print('hi touch 2 ->', t0, hiTouch, close.loc[hiTouch])
        else:
           # display('LOW')
            t_touch = -1
            r_ret = close.loc[lowTouch]
            _touchidx = lowTouch
            #print('low touch 2 ->', t0, lowTouch, close.loc[lowTouch])

            
        ret.loc[t0].touch_tp = t_touch * side[t0]
        ret.loc[t0].raw_return = abs(r_ret - close[t0]) * ret.loc[t0].touch_tp
        if isPercentUse == True:
            ret.loc[t0].p_return = abs(r_ret / close[t0] - 1 ) * ret.loc[t0].touch_tp
        ret.loc[t0].side = side[t0]
        ret.loc[t0].cross_idx = _touchidx

        
    return ret
    
def metaLabling_executer(df, col='close', colEvents='dir', colSide='dir', verticalBarrier = None, SL = None, TP = None, isPercentUse=False):

    def getVerticalBarriers(_idxs, _wsize=2):
        idxs_len = len(_idxs)
        last_index = _idxs[-1]
        rl = []
        for i in range(idxs_len):
            if(i+_wsize < idxs_len):
                rl.append(_idxs[i+_wsize-1])
            else:
                rl.append(last_index)
        return rl

    def calc_diff(a,b):
        ret = 0.0
        if( ((a >= 0.0) and (b >= 0.0)) or ((a >= 0.0) and (b >= 0.0))):
            ret = abs(a-b)
        else:
            ret = abs(a)+abs(b)

        return ret


    g = df.copy()
    g = g.set_index('Nr')
    f = g[colSide] != 0.0
    close = g[col]
    side = g[f][colSide]
    startEvents = g[f][colEvents].index


    if verticalBarrier is None:
        _verticalBarrier = pd.Series(close.index[-1], index=startEvents)
    else:
        _verticalBarrier = pd.Series(getVerticalBarriers(close.index.values, verticalBarrier), index=close.index)[startEvents]

    if TP is None:
        TP = 10.0

    if SL is None:
        SL = 10.0

    if isPercentUse == True:
        ret = pd.DataFrame(columns=['touch_tp', 'raw_return', 'p_return', 'side', 'cross_idx'], index=startEvents)
    else:
        ret = pd.DataFrame(columns=['touch_tp', 'raw_return', 'side', 'cross_idx'], index=startEvents)
    t_touch = 0
    r_ret = 0
    #p_ret = 0
    #print('startEvents',startEvents)
    #for t0 in tqdm(startEvents):
    for t0 in startEvents:
        window=close.loc[t0:_verticalBarrier[t0]]
        if isPercentUse == True:
            if(side[t0] == 1):
                trgHi = close[t0] * TP
                trgLow = close[t0] * SL
            else:
                trgHi = close[t0] * SL
                trgLow = close[t0] * TP
        else:
            if(side[t0] == 1):
                trgHi = TP #close[t0] + TP
                trgLow = SL #close[t0] - SL
            else:
                trgHi = SL #close[t0] - SL
                trgLow = TP #close[t0] + TP

        #print('\n_verticalBarrier[t0]',_verticalBarrier[t0])
        #print('trgHi', trgHi)
        #print('trgLow', trgLow)
        
        #print('close[t0]', close[t0])
        #print('(trgHi+close[t0])', (trgHi+close[t0]))
        #print('(close[t0]-trgLow)', (close[t0]-trgLow))
        fabove = window >= (trgHi+close[t0])
        fbelow = window <= (close[t0]-trgLow)
        
        hiTouch = window[fabove].index.min()
        lowTouch = window[fbelow].index.min()

        #print('t0', t0)
        #print('ht',(hiTouch))
        #print('lt',(lowTouch))
        _touchidx = 0
        
        
        
        if ((pd.isnull(hiTouch)) and (pd.isnull(lowTouch))):
            #print('vertical barrier or infinit')
            t_touch = 0
            r_ret = close.loc[_verticalBarrier[t0]]
            #print('v_barrier ->', t0, _verticalBarrier[t0], close.loc[_verticalBarrier[t0]], close[t0])
            
            _touchidx = _verticalBarrier[t0]
            #print('diff',calc_diff(close.loc[_touchidx], close[t0]))
            
        elif (pd.isnull(hiTouch) and (not pd.isnull(lowTouch))):
            #display('low touch')
            t_touch = -1 
            r_ret = close.loc[lowTouch]
            _touchidx = lowTouch
            #print('low touch ->', t0, lowTouch, close.loc[lowTouch])
        elif (not pd.isnull(hiTouch) and (pd.isnull(lowTouch))):
            #display('hi touch')
            t_touch = 1 
            r_ret = close.loc[hiTouch]
            _touchidx = hiTouch
            #print('hi touch ->', t0, hiTouch, close.loc[hiTouch])
        elif(hiTouch < lowTouch):
            #display('HI')
            t_touch = 1
            r_ret = close[hiTouch]
            _touchidx = hiTouch
            #print('hi touch 2 ->', t0, hiTouch, close.loc[hiTouch])
        else:
           # display('LOW')
            t_touch = -1
            r_ret = close.loc[lowTouch]
            _touchidx = lowTouch
            #print('low touch 2 ->', t0, lowTouch, close.loc[lowTouch])

        #print('r_ret',r_ret)
                

        ret.loc[t0].touch_tp = t_touch * side[t0]
        if(0.0 == abs(ret.loc[t0].touch_tp)):
            ret.loc[t0].raw_return = calc_diff(close.loc[t0],close.loc[_touchidx])
        else:
            ret.loc[t0].raw_return = calc_diff(close.loc[t0],close.loc[_touchidx])* ret.loc[t0].touch_tp

        #print('diff',calc_diff(close.loc[t0],close.loc[_touchidx]))
        #print('touch_tp', ret.loc[t0].touch_tp)

        if isPercentUse == True:
            print('!!')
            ret.loc[t0].p_return = abs(r_ret / close[t0] - 1 ) * ret.loc[t0].touch_tp
        
        
        ret.loc[t0].side = side[t0]
        ret.loc[t0].cross_idx = _touchidx



    #print(ret.loc[:,'touch_tp'])
    ret.loc[:,'touch_tp']= ret.loc[:,'touch_tp'].replace([-1], -1.0)
    ret.loc[:,'touch_tp']= ret.loc[:,'touch_tp'].replace([0], 255)

    ret = pd.merge(g, ret[['touch_tp','cross_idx','raw_return']], how='left', left_index=True, right_index=True)
    #print(g1.head())
    ret.loc[:,'touch_tp'] = ret.loc[:,'touch_tp'].fillna(0.0)
    ret.loc[:,'cross_idx'] = ret.loc[:,'cross_idx'].fillna(0.0)

        #ret = ret.set_index('date')

    return ret


#generate labels with multiprocessing
def metalabeling_labels_mp(df, col='close', colEvents='dir', colSide='dir', grpby='date', verticalBarrier = None, SL = None, TP = None, isPercentUse=False):
    
    grp_1 = df.groupby(grpby)

    
    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
        results = pool.starmap(metaLabling_executer, 
                               [(g, col, colEvents, colSide, verticalBarrier, SL, TP, isPercentUse) for n,g in grp_1])
        
    results_df = pd.concat(results)
    #results_df = results_df.reset_index().set_index('date')
    
    return results_df    




def ret_rnn_value(data, indices, history_size, cur_index=None, target_size=0):
    return np.reshape(data[indices], (history_size, 1))

def ret_rnn_label(data, indices, history_size, cur_index=None, target_size=0):
    return data[cur_index+target_size]

def get_rnn_data_for_one_col(values, start_index, end_index, history_size, fn, target_size=0, isPadding=False):

    if isPadding == True:
        if(isinstance(values[0], np.datetime64)):
          #zr = numpy.array([value[0] - datetime.timedelta(hours=i) for i in xrange(24)])
          zr = np.array([values[0] for i in range(history_size-1)])
        else:
          zr = np.zeros(history_size-1)
        workset = np.append(zr, values)
    else:
        workset = values.copy()

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(values) - target_size
    data = []
    for i in range(start_index, end_index):
        indices = range(i-history_size+1, i+1)
        data.append(fn(workset, indices, history_size, i, target_size))
    return np.array(data)

def _create_rnn_data(dataset, start_index, end_index, history_size, target_col, index_col, used_col, isPadding=False):
    #ds = dataset[start_index:end_index].copy()
    ds_col = list(dataset[used_col].columns.values)


    res = {}
    res_df = pd.DataFrame()

    # Xs
    for cl in ds_col:
        res[cl] = get_rnn_data_for_one_col(dataset[cl].values, start_index, end_index, history_size, ret_rnn_value, isPadding=isPadding)
        res[cl] = pd.Series(res[cl].tolist())
        res_df[cl] = res[cl]
        res_df[cl] = res_df[cl].apply(lambda x: np.array(x))
        #print('res_df[cl].shape',res_df[cl].shape)
    
    # Labels
    for cl in target_col:
      lb = get_rnn_data_for_one_col(dataset[cl].values, start_index, end_index, history_size, ret_rnn_label,isPadding=isPadding)
      res_df[cl] = lb

    #set index
    for cl in index_col:
      idx = get_rnn_data_for_one_col(dataset[cl].values, start_index, end_index, history_size, ret_rnn_label,isPadding=isPadding)
      res_df[cl] = idx


    return res_df


def create_rnn_data_mp(dataset, start_index, end_index, history_size, target_col, index_col, grby='date', used_col=[], isPadding=False):
  
    ds_col = list(dataset.loc[:, ~dataset.columns.isin([grby])].columns.values)

    _grp = dataset.groupby(grby)

    with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
        results = pool.starmap(_create_rnn_data, 
                                [(g, start_index, end_index, history_size, target_col, index_col, used_col, isPadding) for n,g in _grp])
        
    results_df = pd.concat(results)

 
    return results_df
