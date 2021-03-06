'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import parser
import time
import os
import numpy as np
import gym
import logz
import ray
import utils
import optimizers
from reward_func import *
from policies import *
import socket
from shared_noise import *
from utils import Logger
from random import shuffle 

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        self.env = gym.make(env_name)
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):

        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)
        
        env = gym.make(env_name)
        
        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params

        self.env_name = env_name
        self.seed = seed

        # exp statistics
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        
        # create shared table for storing noise
        print("Creating deltas table.")
        # generate noise sequence of fixed size in object stores
        deltas_id = create_shared_noise.remote()

        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]


        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

        # params->return tuple dataset
        self.dataset_x = []
        self.dataset_y = []
        self.batch_size = 2*self.num_deltas # set batch_size equal to num_directions
        self.reward_func = RewardFunction(params_dim= self.w_policy.size, hidden_dim= 100, lr= 1e-2, seed= self.seed)


    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        # int(5/2) = 2 向下取整
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], [] 

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards


        # append data into train dataset
        for i in range(deltas_idx.size):
            delta = self.deltas.get(deltas_idx[i], self.w_policy.size)
            delta = (self.delta_std * delta).reshape(self.w_policy.shape)

            params_pos = (self.w_policy + delta).flatten()
            params_neg = (self.w_policy - delta).flatten()

            reward_pos = rollout_rewards[i, 0]
            reward_neg = rollout_rewards[i, 1]
            # self.dataset.append((params_pos, reward_pos))
            # self.dataset.append((params_neg, reward_neg))
            self.dataset_x.append(params_pos)
            self.dataset_y.append(reward_pos)
            self.dataset_x.append(params_neg)
            self.dataset_y.append(reward_neg)


        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]


        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat = self.aggregate_rollouts()                    
        # self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)

        # set train hyparameter for reward function learning 
        train_size = len(self.dataset_x)
        reward_avg_loss = 0.0
        gradient_mean = np.zeros(self.w_policy.shape)

        print("train dataset size:", train_size)
        assert len(self.dataset_x) == len(self.dataset_y)

        if(train_size > 20):
            reward_train_epochs = 10000
            dataset_x_array = np.array(self.dataset_x)
            dataset_y_array = np.array(self.dataset_y).reshape(-1,1)
            # print(dataset_x_array.shape)
            # print(dataset_y_array.shape)

            self.batch_size = train_size
            num_batch = int(train_size / self.batch_size)

            # output reward loss before train
            loss_list = []
            for i in range(num_batch):
                loss_list.append(self.reward_func.sess.run(self.reward_func.loss, feed_dict={self.reward_func.input_ph: dataset_x_array[i*self.batch_size: (i+1)*self.batch_size], 
                                    self.reward_func.reward_ph: dataset_y_array[i*self.batch_size: (i+1)*self.batch_size]}))
            # print(loss_list)
            reward_avg_loss = np.mean(loss_list)
            print("reward_avg_loss_before:", reward_avg_loss)

            # batch_size = min(256, train_size)
            batch_size = train_size
            # train reward function
            for i in range(reward_train_epochs):
                # init_list = [i for i in range(train_size)]
                # shuffle(init_list)
                # shuffle_dataset_x = dataset_x_array[init_list]
                # shuffle_dataset_y = dataset_y_array[init_list]
                # self.reward_func.sess.run(self.reward_func.train_op, feed_dict={self.reward_func.input_ph: shuffle_dataset_x[0:batch_size], 
                #                     self.reward_func.reward_ph: shuffle_dataset_y[0:batch_size]})
                self.reward_func.sess.run(self.reward_func.train_op, feed_dict={self.reward_func.input_ph: dataset_x_array[0:batch_size], 
                                    self.reward_func.reward_ph: dataset_y_array[0:batch_size]})
                if(i%3000 == 0) and (i != 0):
                    loss_list = []
                    for j in range(num_batch):
                        loss_list.append(self.reward_func.sess.run(self.reward_func.loss, feed_dict={self.reward_func.input_ph: dataset_x_array[j*batch_size: (j+1)*batch_size], 
                                            self.reward_func.reward_ph: dataset_y_array[j*batch_size: (j+1)*batch_size]}))
                    reward_avg_loss = np.mean(loss_list)
                    print("epoch {0}: reward_avg_loss:{1}".format(i, reward_avg_loss))

            # output reward function loss
            loss_list = []
            for i in range(num_batch):
                loss_list.append(self.reward_func.sess.run(self.reward_func.loss, feed_dict={self.reward_func.input_ph: dataset_x_array[i*self.batch_size: (i+1)*self.batch_size], 
                                    self.reward_func.reward_ph: dataset_y_array[i*self.batch_size: (i+1)*self.batch_size]}))
            reward_avg_loss = np.mean(loss_list)
            print("reward_avg_loss_after:", reward_avg_loss)

            # generate gradient from reward function
            reward_sample_num = self.batch_size
            reward_directions = []
            for i in range(reward_sample_num):
                idx, delta = self.deltas.get_delta(self.w_policy.size)
                delta = (self.delta_std * delta).reshape(self.w_policy.shape)
                params = (delta + self.w_policy).flatten()
                reward_directions.append(params)

            params_gradient = self.reward_func.sess.run(self.reward_func.grad_x, feed_dict={self.reward_func.input_ph: reward_directions[0:self.batch_size]})
            # params_gradient = grad_var_list[0][0]
            params_gradient = np.asarray(params_gradient)
            # print(params_gradient.shape)
            gradient_mean = np.mean(params_gradient, axis=(0,1))
            # print(gradient_mean.shape)

        beta = 0.5
        func_lr_mul = 1e-2
        gradient_mean = gradient_mean * func_lr_mul
        print("Euclidean norm of ES update:", np.linalg.norm(g_hat))
        print("Euclidean norm of function gradient:", np.linalg.norm(gradient_mean)) 
        # step = -self.stepsize * globalg
        # self.w_policy += beta * func_lr * gradient_mean.reshape(self.w_policy.shape)
        self.w_policy -= beta * self.optimizer._compute_step(gradient_mean).reshape(self.w_policy.shape)
        self.w_policy -= (1 - beta) * self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)

        return reward_avg_loss

    def train(self, num_iter):

        log_name = "seed_{0}".format(self.seed)
        logger = Logger(logname= self.env_name, now= log_name)

        start = time.time()
        for i in range(num_iter):
            
            t1 = time.time()
            reward_avg_loss = self.train_step()
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')

            # record statistics every 10 iterations
            if ((i + 1) % 10 == 0):
                
                rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/lin_policy_plus", w)
                
                # # output reward function loss
                # test_loss_list = []
                # test_size = len(test_dataset_x)
                # assert len(test_dataset_x) == len(test_dataset_y)
                # test_dataset_x = np.array(test_dataset_x)
                # test_dataset_y = np.array(test_dataset_y).reshape(-1,1)
                # num_batch = int(test_size / self.batch_size)

                # for idx in range(num_batch):
                #     test_loss_list.append(self.reward_func.sess.run(self.reward_func.loss, feed_dict={self.reward_func.input_ph: test_dataset_x[idx*self.batch_size: (idx+1)*self.batch_size], 
                #                                                     self.reward_func.reward_ph: test_dataset_y[idx*self.batch_size: (idx+1)*self.batch_size]}))
                # test_avg_loss = np.mean(test_loss_list)
                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.log_tabular("AvgRewardFunctionLoss", reward_avg_loss)
                # logz.log_tabular("AvgRewardTestLoss", test_avg_loss)
                logz.dump_tabular()
                
                logger.log({"Time": time.time() - start,
                    "Iteration": i + 1,
                    "AverageReward": np.mean(rewards),
                    "StdRewards": np.std(rewards),
                    "MaxRewardRollout": np.max(rewards),
                    "MinRewardRollout": np.min(rewards),
                    "timesteps": self.timesteps
                    })
                logger.write(display=False)

            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
        
        np.savetxt("dataset_x.txt", self.dataset_x)
        np.savetxt("dataset_y.txt", self.dataset_y)
        logger.close()    
        return 

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'])
        
    ARS.train(params['n_iter'])
       
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(redis_address= local_ip + ':6379')
    ray.init()
    
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)
    # run_ars(**vars(args)) 解析args_dict

