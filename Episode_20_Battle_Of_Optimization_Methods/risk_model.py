import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_probability as tfp

def pct_change(nparray):
    pct=np.zeros_like(nparray)
    a = nparray.T
    a = np.diff(a) / a[:,:-1] 
    pct[1:] = a.T
    return pct

def scale_0_1(data):
    return (data-np.min(data, axis=0))/(np.max(data, axis=0)-np.min(data, axis=0)+0.000000001)

class Model:
    def __init__(self, rnd_seed = 0):
        self.data = None
        self.model = None
        tf.random.set_seed(rnd_seed)
        np.random.seed(rnd_seed)
    
    def getModel(self):
        return self.model

    def resetModel(self):
        self.model = None
        
    def __build_model(self, input_shapes, outputs):
        
        @tf.function
        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
                
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe


        model = Sequential([
            LSTM(64,  input_shape=input_shapes),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])
 
        model.compile(loss=sharpe_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
        return model

    def get_allocations(self, data, **params):
    # data with returns
        data_w_ret = data_w_ret = np.concatenate([ scale_0_1(data[1:]), (pct_change(data))[1:] ], axis=1)
        
        data = data[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, data.shape[1])
        
        fit_predict_data = data_w_ret[np.newaxis,:]     

        callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)   
        self.model.fit(fit_predict_data, np.zeros((1, data.shape[1])), epochs=params['epochs'], 
        shuffle=False,callbacks=[callback_early_stop], verbose=0)
        return self.model.predict(fit_predict_data)[0]

