from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
import os
from collections import deque
from pandas_datareader import data
import random
import datetime

from tf_agents.environments import py_environment
#from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from sklearn.preprocessing import MinMaxScaler


#temoparay here
def mprint(_s, verbose = 1):
    if(verbose == 1):
        print(_s)

verbose = 0

POS_LONG = 1
POS_SHORT = -1
POS_STATE_OPEN = 'open'
POS_STATE_CLOSE = 'close'
POS_STATE_SKIP= 'skip'
#enviroment
class MyTradeEnv(py_environment.PyEnvironment):

    def __getRandomDataFrom(self, stocks, data_range, verbose=0):
        ''' function reads data from the yahoo finance and returns price data '''
        ticker = random.choice(stocks)
        if(verbose == 1):
            print('selected stock', ticker)
        
        stock = data.DataReader(ticker, 
                        start=data_range['start'],
                        end=data_range['end'],
                        data_source='yahoo')
        stock.reset_index(drop=True, inplace=True)
        stock = stock.drop(['Adj Close'], axis=1)
        #temp
        #stock = stock.iloc[0:6]
        return ticker, stock

    # function return observation data, according to the current index
    def __get_observation_data(self, idx):
        #print(idx, self._logPriceArray.shape)
        return self._logPriceArray.iloc[idx]

    def __get_price_data(self, idx):
        col_name = 'Open'
        # select close price if we are usin Market-on-Close
        if( True == self._isMOCUsed ):
            col_name = 'Close'
        return self._priceArray.iloc[idx][col_name]

    # used for update enviroment observation step
    # to updating observation step, only this function should be used
    def __update_state(self, _idx):
        # state is a dictionary wich consist of price data and position state
        self._state = {
            'price' : self.__get_observation_data(_idx).values.astype(np.float32),
            'pos' : self._activePosition.astype(np.float32)
            }

    # used for select initial episode
    # we use only n prices, we will randomly select a price slice
    def __select_episode(self):

        if(self.change_stock_counter > self.change_stock_step):
            self.change_stock_counter = 0
            self.ticker, self.price_data = self.__getRandomDataFrom(self.stocks, self.data_range, verbose=1)

        self.change_stock_counter = self.change_stock_counter +1 

        #select random start index
        if(self.price_data.shape[0] > self.minReqiredData):
            self._start_index_of_obs_data = np.random.randint(self.price_data.shape[0] - self.minReqiredData, size=1)[0]
        else:
            self._start_index_of_obs_data = 0
        pa = self.price_data.iloc[self._start_index_of_obs_data:]
        logpa = np.log1p(pa.pct_change()).dropna().fillna(0.0)
        #logpa = np.log1p(pa.pct_change()).cumsum()
        pa = pa.iloc[1:]
        #print('log, pa', logpa.shape, pa.shape)
        return pa, logpa

    def __isPositionOpened(self):
        return True if(self._position != 0) else False
        
    def __calcShareNumber(self, maxBP, price):
        return int(maxBP / price)

    def __getAvaliableBP(self):
        return self._money * self._maxPctCapital

    def __open_position(self, pos, action, price_index):
        
        reward = 0
        actionPnl = 0

        if(False == self.__isPositionOpened()): #no open positions

            self._position = pos # set position direction
            #update the position data
            if(pos == POS_LONG):
                self._activePosition[0] = 1.0
            elif(pos == POS_SHORT):
                self._activePosition[1] = 1.0

            # shares = self.__calcShareNumber(self.__getAvaliableBP(), self.__get_price_data(price_index))
            # print('shares', shares)
            needMoney = self._shares * self.__get_price_data(price_index) #calculate how much we need to pay

            if(needMoney > self._money): #no more money to trade => exit
                self._episode_ended = True # fnish the episode 
                reward = -100  
                mprint('ERROR NO MONEY!-> needMoney = {}, shares = {}, price = {}'.format(needMoney, self._shares, self.__get_price_data(price_index)), verbose=verbose)
            else:  #open new position
                actionPnl = self.__calculatePNL(pos_state=POS_STATE_OPEN, price_index=price_index) # self._fees - self._spread*self._shares # calculate immidiade pnl
                
               # mprint('PRICE: {}, {}'.format(price_index, self._priceArray),verbose=verbose)
                mprint('OPEN -> actionPnl = {}, price = {}'.format(actionPnl,  self.__get_price_data(price_index)), verbose=verbose)
                #reward = 0
                reward = actionPnl #set reward equal pnl for current day

        else: # position is already open
            action = 0 #set action to skip
            mprint('OPEN LONG -> Position Open -> go to Skip', verbose=verbose)


        return reward, actionPnl, action
    

    def __reset_position(self):
        self._position = 0
        self._activePosition = np.array([0,0]).astype(np.int32)


    def __calculatePNL(self, pos_state, price_index, isPercentage = True):
        ret = 0
        #print('---------------->', pos_state)
        mprint('ISPERSANTAGE: {}'.format(isPercentage),verbose=verbose)
        mprint('price_index: {}'.format(price_index),verbose=verbose)
        if pos_state == POS_STATE_OPEN:
            if isPercentage == True:
                ret = ((self._fees - self._spread) * 100) / self.__get_price_data(price_index)
            else:
                ret = self._fees - self._spread*self._shares

        elif pos_state == POS_STATE_SKIP:
            diffPrice = self.__calculate_price_change(self._currentIndex, self._currentIndex-1)
            if isPercentage == True:
                ret = ((diffPrice) * 100) / self.__get_price_data(self._currentIndex-1)
            else:
                ret = diffPrice * self._shares


        elif pos_state == POS_STATE_CLOSE:
            diffPrice = self.__calculate_price_change(self._currentIndex, self._currentIndex-1)
            if isPercentage == True:
                ret = ((diffPrice + self._fees - self._spread) * 100) / self.__get_price_data(self._currentIndex-1)
            else:
                ret = diffPrice * self._shares + self._fees - self._spread*self._shares
        #print('<---------------- ret', ret )
        return ret

    def __calculate_price_change(self, icur, iprev):
        return (np.abs(self.__get_price_data(icur)) - np.abs(self.__get_price_data(iprev)))*self._position

    #################################################################
    #################################################################
    #################################################################
    def __init__(self, stocks, data_range, isMOC = True):
        
        self.stocks = stocks
        self.data_range = data_range


        #define number when we will change the stock
        self.change_stock_step = 1000
        self.change_stock_counter = 0

        #used for idndication of the end of episode        
        self._episode_ended = False

        self._position = 0      # 0 - no position, 1 - long, -1 - short

        self._start_index_of_obs_data = 0
        
        #additional configuration data
        self._fees = 0#-0.005       # broker fees
        self._spread = 0#0.02     # spread
        self._shares = 100      # number of trading shares
        self._money = 20000.0   # money
        self._pnl = 0.0         # PnL
        self._maxPnl = 0.0      # maximun pnl 
        self._maxPctCapital = 20.0 # maximum capital for one trade
        self._maxPctLoss = 5.0 # maximum cumulativ loss %

        self._currentIndex = 0       # days of history are available from the beginning

        self.minReqiredData = 30

        self._isMOCUsed = isMOC # True - MOC will be used, False MOO - used
        self.ticker = ""
        #getting price data
        self.ticker, self.price_data = self.__getRandomDataFrom(self.stocks, self.data_range)

        #_priceArray - array for price calculation
        self._priceArray, self._logPriceArray = self.__select_episode()

        # active position -> onehot encoded -> index 0 = 1 - open long position, index 1 = 1 - open short position
        self._activePosition = np.array([0,0])

        #calculate new observation state
        #every new step (new day) we are adding new price
        self.__update_state(self._currentIndex)

        #3 Actions: We have 3 actions. Action 0: skip, Action 1: buy, Action 2: sell, Action 3: close open position
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        #representation of the enviroment: price + open position state
        self._observation_spec = {
            'price':array_spec.BoundedArraySpec(shape=(self._logPriceArray.shape[1], ), dtype=np.float32, name='obs_price'),
            'pos':array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0, name='obs_pos')
        }

        mprint('PRICE: {}'.format(self._priceArray),verbose=verbose)
       
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._currentIndex = 0
        self._pnl = 0.0
        self._maxPnl = 0.0

        self._priceArray, self._logPriceArray = self.__select_episode()
        self.__reset_position()
        self.__update_state(self._currentIndex)

        #return ts.restart(np.array(self._state, dtype=np.float32))
        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        reward = 0
        actionPnl = 0

        # action selected on visible history, par performs on the next observation
        mprint('action = {} _idx = {}'.format(action, self._currentIndex), verbose=verbose)

        #NOTE: currently on MOC is supported
        #price index should be close of previous state, because we already update enviroment
        #select the action 
        if(1 == action): #buy
            reward, actionPnl, action = self.__open_position(POS_LONG, action, self._currentIndex)

        elif(2 == action): #sell
            reward, actionPnl, action = self.__open_position(POS_SHORT, action, self._currentIndex)

        elif(3 == action): #close
            if(self._position != 0):
                #calculate the pirce change
                actionPnl = self.__calculatePNL(pos_state='close', price_index=0)
                mprint('CLOSE -> actionPnl = {}, price = {}'.format(actionPnl,  self.__get_price_data(self._currentIndex)), verbose=verbose)

                self.__reset_position()
                reward = actionPnl #set reward equeal to day pnl
            else:
                action = 0 # nothing to close -> go to skip action


        if(0 == action): #skip action 
            if(self._position != 0): #position is open
                actionPnl = self.__calculatePNL(pos_state='skip', price_index=0)
                mprint('SKIP -> actionPnl = {}, price = {}'.format(actionPnl,  self.__get_price_data(self._currentIndex)), verbose=verbose)
                reward = actionPnl
            else:
                mprint('SKIP -> ...no position', verbose=verbose)
                reward = 0 # if no position set negative reward to stimulate agent

        #print(self._currentIndex, self._priceArray.shape[0]-1, self._logPriceArray.shape)
        if self._currentIndex == (self._priceArray.shape[0]-1): #if we have a last observation day
            self._episode_ended = True


        if self._episode_ended == True:
            if(True == self.__isPositionOpened()):
                actionPnl = self.__calculatePNL(pos_state='close', price_index=0)
                mprint('EP END -> CLOSE -> actionPnl = {}, price = {}'.format(actionPnl,  self.__get_price_data(self._currentIndex)), verbose=verbose)
                self.__reset_position()
                reward = actionPnl
            else:
                mprint('EP END -> _pnl = {}'.format(self._pnl), verbose=verbose)
        else:
            self._currentIndex += 1 #go to the next day
            self.__update_state(self._currentIndex) #update observation

        self._pnl += actionPnl # update PnL
        mprint('PnL = {}'.format(self._pnl), verbose=verbose)


        if self._pnl > self._maxPnl:
            self._maxPnl = self._pnl
        if self._pnl < (self._maxPnl-self._maxPctLoss):
            self._episode_ended = True
            reward = -100 #- self._pnl
            #print('pnl:', self._pnl, 'maxPnl:', self._maxPnl)


        #print('-->reward', reward)
        if self._episode_ended:
            #reward = actionPnl #self._pnl
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)


