"""
Experience reuse Evolution Stragies

Written by Guoqing (fiberleif.github.io)
"""
import numpy as np
import tensorflow as tf

class RewardFunction(object):
    """ NN-based reward function approximation """
    # def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, seed, seed_op, clipping_range=None):
    def __init__(self, params_dim, hidden_dim, batch_size, lr, seed):

        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
        """
        self.lr = lr
        # self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.params_dim = params_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self._build_graph(seed)
        self._init_session()

    def _build_graph(self, seed):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(seed)
            self._placeholders()
            self._policy_nn()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        # self.input_var = tf.Variable(np.zeros([self.batch_size, self.params_dim]))
        # self.reward_var = tf.Variable(np.zeros([self.batch_size, 1]))
        self.input_ph = tf.placeholder(tf.float32, (None, self.params_dim), 'params')
        self.reward_ph = tf.placeholder(tf.float32, (None, 1), 'rewards')
        # self.lr_ph = tf.placeholder(tf.float32, (), 'eta')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        with tf.variable_scope("reward_params") as scope:
        
            self.h1 = tf.layers.dense(self.input_ph, self.hidden_dim, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.params_dim)), name="h1")
            self.h2 = tf.layers.dense(self.h1, self.hidden_dim, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.params_dim)), name="h2")
            self.rewards = tf.layers.dense(self.h2, 1,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / self.hidden_dim)), name="rewards")
            self.rewards_sum = tf.reduce_sum(self.rewards)

        


    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """

        self.reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="reward_params")
        print(self.reward_params)
        # self.ph_loss = tf.reduce_mean(tf.pow(self.ph_rewards - self.reward_ph, 2))
        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.reward_ph, 2))

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.reward_params)
        self.grad_x = tf.gradients(self.rewards, self.input_ph)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)


    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
