from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
import os

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

#enviroment
class MyTradeEnv(py_environment.PyEnvironment):

    # used for update enviroment observation step
    # to updating observation step, only this function should be used
    def __update_state(self, _idx):
        # state is a dictionary wich consist of price data and position state
        self._state = {
            'price' : self.__get_observation_data(_idx).astype(np.float32),
            'pos' : self._activePosition.astype(np.int32)
            }

    # function return observation data, according to the current index
    def __get_observation_data(self, idx):
        _x = MinMaxScaler().fit_transform(self._selectedData).astype(np.float32)
        _x[idx:] = 0
        return _x

    # used for select initial episode
    # we use only n prices, we will randomly select a price slice
    def __select_episode(self, idx):
        #select random start index
        self._start_index_of_obs_data = np.random.randint(self.price_data.shape[0] - self._maxDayIndex -1, size=1)[0]
        #self._start_index_of_obs_data = np.random.randint(5, size=1)[0]
        #self._start_index_of_obs_data = 1
        #secelct observation price data
        na = self.price_data.iloc[self._start_index_of_obs_data:self._start_index_of_obs_data + self._maxDayIndex ].values.astype(np.float32)
        #select price array for pnl calculation
        pa = self.__get_price_data_for_pnl_calculation(idx)

        return pa, na

    def __get_price_data_for_pnl_calculation(self, idx):
        col_name = 'Open'
        # select close price if we are usin Market-on-Close
        if( True == self._isMOCUsed ):
            col_name = 'Close'
        pa = self.price_data.iloc[self._start_index_of_obs_data:self._start_index_of_obs_data + self._maxDayIndex ][col_name].values
        return pa[idx-3:idx].astype(np.float32)


    def __init__(self, df, isMOC = True):
        #2 Actions: We have 3 actions. Action 0: skip, Action 1: buy, Action 2: sell, Action 3: close open position
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        #representation of the enviroment: price + open position state
        self._observation_spec = {
            'price':array_spec.BoundedArraySpec(shape=(20,5), dtype=np.float32, minimum=0, name='obs_price'),
            'pos':array_spec.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=0, maximum=1, name='obs_pos')
        }


        #define price data
        self.price_data = df
        #used for idndication of the end of episode        
        self._episode_ended = False

        self._position = 0      # 0 - no position, 1 - long, -1 - short

        self._start_index_of_obs_data = 0
        
        #additional configuration data
        self._fees = -0.5       # broker fees
        self._spread = 0.02     # spread
        self._shares = 100      # number of trading shares
        self._money = 20000.0   # money
        self._pnl = 0.0         # PnL

        self._dayIndex = 10       # days of history are available from the beginning
        self._maxDayIndex = 20    # maximun days of trading investment
        self._currentDayIndex = 0 # current day idex -> starts with 0

        
        self._isMOCUsed = isMOC # True - MOC will be used, False MOO - used
        
        #_priceArray - array for price calculation
        self._priceArray, self._selectedData = self.__select_episode(self._dayIndex)

        # active position -> onehot encoded -> index 0 = 1 - open long position, index 1 = 1 - open short position
        self._activePosition = np.array([0,0])

        #calculate new observation state
        #every new step (new day) we are adding new price
        self.__update_state(self._dayIndex)
       
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._currentDayIndex = 0
        self._pnl = 0

        self._priceArray, self._selectedData = self.__select_episode(self._dayIndex)
        self.__reset_position()
        self.__update_state(self._dayIndex)

        #return ts.restart(np.array(self._state, dtype=np.float32))
        return ts.restart(self._state)

    def __open_position(self, pos, action, price_index):
        
        reward = 0
        actionPnl = 0

        if(self._position == 0): #no open positions

            self._position = pos # set position direction
            #update the position data
            if(pos == 1):
                self._activePosition[0] = 1.0
            elif(pos == -1):
                self._activePosition[1] = 1.0

            needMoney = self._shares * self._priceArray[price_index] #calculate how much we need to pay

            if(needMoney > self._money): #no more money to trade => exit
                self._episode_ended = True # fnish the episode 
                reward = -100  
                mprint('ERROR NO MONEY!-> needMoney = {}, shares = {}, price = {}'.format(needMoney, self._shares, self._priceArray[price_index]), verbose=verbose)
            else:  #open new position
                actionPnl = self._fees - self._spread*self._shares # calculate immidiade pnl
                mprint('PRICE: {}, {}'.format(price_index, self._priceArray),verbose=verbose)
                mprint('OPEN -> actionPnl = {}, price = {}'.format(actionPnl,  self._priceArray[price_index]), verbose=verbose)
                #reward = 0
                reward = actionPnl #set reward equal pnl for current day

        else: # position is already open
            action = 0 #set action to skip
            mprint('OPEN LONG -> Position Open -> go to Skip', verbose=verbose)


        return reward, actionPnl, action
    

    def __reset_position(self):
        self._position = 0
        self._activePosition = np.array([0,0]).astype(np.int32)


    def __calculate_price_change(self, i1, i2):
        return (np.abs(self._priceArray[i1]) - np.abs(self._priceArray[i2]))*self._position

    def __calculate_close_position_pnl(self, diffPrice):
        return diffPrice * self._shares + self._fees - self._spread*self._shares

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        reward = 0
        actionPnl = 0

        #go to the next day
        self._currentDayIndex += 1

        # calculate new index, because we can't buy or sell with the visible price
        # action selected on visible history, par performs on the next observation
        _idx = self._dayIndex + self._currentDayIndex
        mprint('action = {} _idx = {}'.format(action,  _idx), verbose=verbose)

        #update observation 
        self.__update_state(_idx)
        #update calculation price array
        self._priceArray = self.__get_price_data_for_pnl_calculation(_idx)

        #NOTE: currently on MOC is supported
        #price index should be close of previous state, because we already update enviroment
        #select the action 
        if(1 == action): #buy
            reward, actionPnl, action = self.__open_position(1, action, 1)

        elif(2 == action): #sell
            reward, actionPnl, action = self.__open_position(-1, action, 1)


        elif(3 == action): #close
            if(self._position != 0):

                #calculate the pirce change
                diffPrice = self.__calculate_price_change(1, 0)
                actionPnl = self.__calculate_close_position_pnl(diffPrice)
                mprint('CLOSE -> actionPnl = {}, price = {}'.format(actionPnl,  self._priceArray[1]), verbose=verbose)

                self.__reset_position()

                reward = actionPnl #set reward equeal to daychange pnl

            else:
                action = 0 # nothing to close -> go to skip action


        if(0 == action): #skip action 
            if(self._position != 0): #position is open
                diffPrice = self.__calculate_price_change(1, 0)
                actionPnl = diffPrice * self._shares
                mprint('SKIP -> actionPnl = {}, price = {}'.format(actionPnl,  self._priceArray[1]), verbose=verbose)
                
                #reward = 0
                reward = actionPnl
            else:
                mprint('SKIP -> ...no position', verbose=verbose)
                reward = -10 # if no position set negative reward to stimulate agent


        if _idx == self._maxDayIndex: #if we have a last observation day
            self._episode_ended = True


        if self._episode_ended == True:
            if(self._position != 0):
                diffPrice = self.__calculate_price_change(1, 0)
                actionPnl = self.__calculate_close_position_pnl(diffPrice)

                self.__reset_position()



        self._pnl += actionPnl # update PnL

        self.__update_state(_idx)

        mprint('PnL = {}'.format(self._pnl), verbose=verbose)
        #mprint('{}'.format(self._state), verbose=verbose)

        #print('-->reward', reward)
        if self._episode_ended:
            reward = actionPnl #self._pnl
            #return ts.termination(np.array(self._state, dtype=np.float32), reward)
            return ts.termination(self._state, reward)
        else:
            #return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)
            return ts.transition(self._state, reward=reward, discount=1.0)

