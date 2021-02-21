from typing import get_args
from flask import Flask
from flask import Response, make_response
from datetime import datetime
import re

import tensorflow as tf
from tf_agents.trajectories import time_step as ts
import numpy as np

app = Flask(__name__)


policy_dir = './policy/'
saved_policy = tf.saved_model.load(policy_dir)
policy_state = saved_policy.get_initial_state(1)


def convertDataToTimeStep(data, step_type):
    
    # scale price data
    st = tf.constant(np.array(np.array([step_type], dtype=np.int32)))
    rw = tf.constant(np.array(np.array([0], dtype=np.float32)))
    ds = tf.constant(np.array(np.array([0], dtype=np.float32)))
    
    ts_obs = {}
    for key, value in data.items():
        ts_obs[key] = tf.expand_dims(tf.convert_to_tensor(value, dtype=np.float32), axis=0)
    
    t = ts.TimeStep(st, rw, ds, ts_obs)

    #print(t)

    return t


def runPolicy(t, _policy, _policy_state):
    if( t.step_type.numpy()[0] == 0):
        _policy_state = _policy.get_initial_state(1)
    #_policy_state = _policy.get_initial_state(1)

    a, _policy_state, _ = _policy.action(t, _policy_state)

    return a, _policy_state


@app.route("/testmodel")
def testmodel():
    global saved_policy, policy_state
    o={
    'price': [[0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.03052988, 0.03897642, 0.03963647, 0.02822814, 0.01129203]],

      'pos': [[0., 0.],
       [0., 0.],
       [0., 0.]]
      }

    t = convertDataToTimeStep(o, 0)
    a, policy_state = runPolicy(t, saved_policy, policy_state)
    return "action: {}".format(str(a.numpy()[0]))

@app.route("/")
def home():
    return "Hello, Flask!"


from flask import request, jsonify
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


@app.route('/predict',methods=['POST'])
def results():
    global saved_policy, policy_state
    data = request.get_json(force=True)
    user_id = 'undefine'
    if('HTTP_APP_ID' in request.headers.environ):
        user_id = request.headers.environ['HTTP_APP_ID']

    print('user_id', user_id)
    print(data)

    t = convertDataToTimeStep(data, 0)
    print(t)
    a, policy_state = runPolicy(t, saved_policy, policy_state)
    print(a)
    a = int(a.numpy()[0])
    print("action",a)
    resp = jsonify({'action':a})
    resp.headers.set('app_id',user_id)

    return resp

