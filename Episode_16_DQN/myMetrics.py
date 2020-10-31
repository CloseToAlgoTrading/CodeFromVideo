from tf_agents.metrics import tf_py_metric
from tf_agents.metrics import py_metric
import tensorflow as tf

class SumOfRewards(py_metric.PyStepMetric):
    def __init__(self, name='SumOfRewards'):
        super(py_metric.PyStepMetric, self).__init__(name)
        self.rewards = []
        self.actions = []
        self.sum_rew = 0.0
        self.reset()
    #add it inside the MaxEpisodeScoreMetric class
    def reset(self):
        self.rewards = []
        self.actions = []
        self.sum_rew = 0.0

    #add it inside the MaxEpisodeScoreMetric class
    def call(self, trajectory):
        if(trajectory.is_first()):
            self.reset()
        
        self.rewards += trajectory.reward
        self.actions += trajectory.action

        if(trajectory.is_last()):      
            print(self.rewards)
            print(self.actions)
            
            
        
    #add it inside the MaxEpisodeScoreMetric class
    def result(self):
        return tf.math.reduce_sum(self.rewards)

class TFSumOfRewards(tf_py_metric.TFPyMetric):

  def __init__(self, name='SumOfRewards', dtype=tf.float32):
    py_metric = SumOfRewards()

    super(TFSumOfRewards, self).__init__(
        py_metric=py_metric, name=name, dtype=dtype)

