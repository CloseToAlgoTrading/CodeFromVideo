from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import os

import tensorflow as tf
from tf_agents.environments import tf_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from tf_agents.metrics import tf_metrics
from myMetrics import TFSumOfRewards

from absl import logging
import datetime
from myMetrics import TFSumOfRewards

import time

class learningHelper:
    def __init__(self, train_env, test_env, agent, global_step, chkpdir='./',
        num_iterations=20000, collect_episodes=100, collect_steps_per_iteration=3000,
        replay_buffer_capacity=20000, batch_size=64, log_interval=500, 
        num_eval_episodes=10, eval_interval = 5, IsAutoStoreCheckpoint = True, collect_policy = None,
        use_tf_functions=True, summaries_flush_secs=10, train_sequence_length = 1, train_checkpoint_interval=5,
        policy_checkpoint_interval=5, summary_interval=5):

        tf.compat.v1.enable_v2_behavior()

        #define collect policy from agent or specific
        if collect_policy is None:
            self.collect_policy = agent.collect_policy
            logging.info('selected agent collect_policy')
        else:
            self.collect_policy = collect_policy
            logging.info('selected USER collect_policy')

        self.train_sequence_length = train_sequence_length
        self.train_checkpoint_interval = train_checkpoint_interval
        self.policy_checkpoint_interval = policy_checkpoint_interval
        self.summary_interval = summary_interval
        self.use_tf_functions = use_tf_functions

        self.IsAutoStoreCheckpoint = IsAutoStoreCheckpoint
        self.num_iterations = num_iterations
        self.collect_episodes = collect_episodes
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity

        self.batch_size = batch_size
        self.log_interval = log_interval

        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        
        self.agent = agent

        self.train_env = train_env
        self.test_env = test_env

        self.global_step = global_step
        
        cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        eval_dir = os.path.join(chkpdir, 'logs/eval/'+cur_time)
        train_dir = os.path.join(chkpdir, 'logs/train/'+cur_time)
        
        # summury writer for train data
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, 
                                                        flush_millis=summaries_flush_secs * 1000)
        self.train_summary_writer.set_as_default()
    
        # summury writer for evaluation
        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, 
                                                        flush_millis=summaries_flush_secs * 1000)

        #metrics for training
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]               
        
        
        #create reply buffer for collection trajactories
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_capacity)
        
        self.collect_driver = dynamic_step_driver.DynamicStepDriver(
                    self.train_env,
                    self.collect_policy,
                    observers=[self.replay_buffer.add_batch] + self.train_metrics,
                    num_steps= self.collect_steps_per_iteration)

        #Checkpointer directories
        self.checkpoint_dir = os.path.join(chkpdir, 'checkpoint')
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.policy_chkp_dir = os.path.join(chkpdir, 'policy_checkpoint')
        Path(self.policy_chkp_dir).mkdir(parents=True, exist_ok=True)
        
        self.policy_dir = os.path.join(chkpdir, 'policy')
        Path(self.policy_dir).mkdir(parents=True, exist_ok=True)
        

        #chackpointer for 
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
            metrics=metric_utils.MetricsGroup(self.train_metrics, 'train_metrics')
            )
        
        self.policy_checkpointer = common.Checkpointer(
            ckpt_dir=self.policy_chkp_dir,
            policy=self.agent.policy,
            global_step=global_step)

        self.tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)
        
        self.local_step_counter = 0
        
        if use_tf_functions:
          # To speed up collect use common.function.
          self.agent.train = common.function(self.agent.train)

            
        logging.info('Initializing is done')
        pass

    
    def evaluate_agent(self, num_eval_episodes = 10):

        eval_metrics = [
            tf_metrics.EnvironmentSteps(),         
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
            tf_metrics.NumberOfEpisodes()
        ]
        
        results = metric_utils.eager_compute(
            eval_metrics,
            self.test_env,
            self.agent.policy,
            num_episodes=num_eval_episodes,
            train_step=self.global_step,
            summary_writer=self.eval_summary_writer,
            summary_prefix='Metrics',
        )

        metric_utils.log_metrics(eval_metrics)
        pass

    def train_agent(self, n_iteration = None):

        local_epoch_counter = 0
        if n_iteration is None:
            n_iteration = self.num_iterations

        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(local_epoch_counter % self.summary_interval, 0)):
        
            observers = [self.replay_buffer.add_batch] + self.train_metrics

            self.replay_buffer.clear()
            time_step = None
            policy_state = self.collect_policy.get_initial_state(self.train_env.batch_size)

            AUTOTUNE = tf.data.experimental.AUTOTUNE
            dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=AUTOTUNE, 
                sample_batch_size=self.batch_size, 
                num_steps=self.train_sequence_length+1).prefetch(AUTOTUNE)

            iterator = iter(dataset)

            def train_step():
                experience, _ = next(iterator)
                return self.agent.train(experience)

            if self.use_tf_functions:
                train_step = common.function(train_step)


            timed_at_step = self.global_step.numpy()
            time_acc = 0

            #training cycle
            for i in range(n_iteration):
                local_epoch_counter = local_epoch_counter + 1
                start_time = time.time()

                self.replay_buffer.clear()
                #collect data
                time_step, policy_state = self.collect_driver.run( time_step=time_step,
                                                              policy_state=policy_state)

                #calculate interation per epoch
                n_steps = int(self.replay_buffer.num_frames().numpy()/self.batch_size)
                #train
                for _ in range(n_steps):
                    train_loss = train_step()


                time_acc += time.time() - start_time


                #if self.global_step.numpy() % self.log_interval == 0:
                if local_epoch_counter % self.log_interval == 0:            
                    print('step = {}, loss = {}'.format( self.global_step.numpy(),
                                                         train_loss.loss))
                    steps_per_sec = (self.global_step.numpy() - timed_at_step) / time_acc
                    print('{} steps/sec'.format(steps_per_sec))
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=self.global_step)
                    timed_at_step = self.global_step.numpy()
                    time_acc = 0

                for train_metric in self.train_metrics:
                    train_metric.tf_summaries(train_step=self.global_step, 
                                              step_metrics=self.train_metrics[:2])

                if local_epoch_counter % self.train_checkpoint_interval == 0:
                    self.train_checkpointer.save(global_step=self.global_step.numpy())                

                if local_epoch_counter % self.policy_checkpoint_interval == 0:
                    self.policy_checkpointer.save(global_step=self.global_step.numpy())    
                    self.save_policy()

                if local_epoch_counter % self.eval_interval == 0:
                    self.evaluate_agent(10)             

                epoch_train_time = time.time() - start_time
                print('Epoch: {}, global_step {}, epoch train time: {}'.format(local_epoch_counter, 
                                                                               self.global_step.numpy(),
                                                                               epoch_train_time ))
            return train_loss


    def get_agent(self):
        return self.agent

    def store_check_point(self):
        self.train_checkpointer.save(self.global_step)
        pass
    def restore_check_point(self):
        self.train_checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()
        pass
    def save_policy(self):
        self.tf_policy_saver.save(self.policy_dir)
        pass

