from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import os

import tensorflow as tf
from tf_agents.environments import tf_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from tf_agents.metrics import tf_metrics
from myMetrics import TFSumOfRewards


from myMetrics import TFSumOfRewards

class learningHelper:
    def __init__(self, train_env, test_env, agent, global_step, chkpdir='./',
        num_iterations=20000, collect_episodes=100, collect_steps_per_iteration=2,
        replay_buffer_capacity=20000, batch_size=64, log_interval=500, 
        num_eval_episodes=10, eval_interval = 5000, verbose = 0, IsAutoStoreCheckpoint = True, collect_policy = None,
        train_sequence_length = 1):

        tf.compat.v1.enable_v2_behavior()
        
        self.verbose = verbose
        self.train_sequence_length = train_sequence_length

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
        if collect_policy is None:
            self.collect_policy = self.agent.collect_policy
            print('selected agent collect_policy')
        else:
            self.collect_policy = collect_policy
            print('selected USER collect_policy')


        self.train_env = train_env
        self.test_env = test_env

        self.global_step = global_step

        #create reply buffer for collection trajactories
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_capacity)

        #Checkpointer
        self.checkpoint_dir = os.path.join(chkpdir, 'checkpoint')
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.policy_dir = os.path.join(chkpdir, 'policy')
        Path(self.policy_dir).mkdir(parents=True, exist_ok=True)

        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
            )

        self.tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)

        pass

    def evaluate_agent(self, n_episodes=100):
        #define metrics
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        average_return = tf_metrics.AverageReturnMetric()
        #rew = TFSumOfRewards()

        #add reply buffer and metrict to the observer
        observers = [num_episodes, env_steps, average_return ]

        _driver = dynamic_episode_driver.DynamicEpisodeDriver(self.test_env, self.agent.policy, observers, num_episodes=n_episodes)

        final_time_step, _ = _driver.run()

        print('eval episodes = {0}: Average Return = {1}'.format(num_episodes.result().numpy(), average_return.result().numpy()))
        return average_return.result().numpy()

    def collect_training_data(self, verbose=0):

        if(verbose > 0):
            #define metrics
            num_episodes = tf_metrics.NumberOfEpisodes()
            env_steps = tf_metrics.EnvironmentSteps()
            #add reply buffer and metrict to the observer
            observers = [self.replay_buffer.add_batch, num_episodes, env_steps]
        else:
            observers = [self.replay_buffer.add_batch]

        self.replay_buffer.clear()
        #create a driver
        #we can create a driver using e.g. random policy
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.train_env, self.collect_policy, observers, num_episodes=self.collect_episodes)

        #collect_steps_per_iteration = 2
        #driver = dynamic_step_driver.DynamicStepDriver(
        #    train_env, tf_policy, observers, num_steps=collect_steps_per_iteration)

        # Initial driver.run will reset the environment and initialize the policy.
        final_time_step, policy_state = driver.run()
        if(verbose > 0):
            #print('final_time_step', final_time_step)
            print('Number of Steps: ', env_steps.result().numpy())
            print('Number of Episodes: ', num_episodes.result().numpy())

        pass 

    def train_step(self, n_steps):
        # Convert the replay buffer to a tf.data.Dataset 
        # Dataset generates trajectories with shape [Bx2x...]
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=AUTOTUNE, 
            sample_batch_size=self.batch_size, 
            num_steps=(self.train_sequence_length + 1)).prefetch(AUTOTUNE)

        iterator = iter(dataset)

        train_loss = None
        for _ in range(n_steps):
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience)            
        
        print('Global steps {}: Traning Loss {}'.format(self.global_step.numpy(), train_loss.loss))

    def train_agent(self, n_epoch):
        for i in range(n_epoch):
            self.collect_training_data(verbose=self.verbose)
            #print('num_frames()', self.replay_buffer.num_frames().numpy())
            #print('n_steps()', int(self.replay_buffer.num_frames().numpy()/self.batch_size))
            self.train_step(int(self.replay_buffer.num_frames().numpy()/self.batch_size))
            if(self.IsAutoStoreCheckpoint == True):
                self.store_check_point()
        pass

    def train_agent_with_avg_ret_condition(self, max_steps, min_avg_return, n_eval_steps=100):
        for i in range(max_steps):
            self.collect_training_data(verbose=self.verbose)
            #print('num_frames()', self.replay_buffer.num_frames().numpy())
            #print('n_steps()', int(self.replay_buffer.num_frames().numpy()/self.batch_size))
            self.train_step(int(self.replay_buffer.num_frames().numpy()/self.batch_size))
            if(self.IsAutoStoreCheckpoint == True):
                self.store_check_point()
            
            if ((i>0) and (i % self.eval_interval) == 0):
                avg_ret = self.evaluate_agent(n_eval_steps)
                if(avg_ret > min_avg_return):
                    return
        pass

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

