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
        self.input_ph = tf.Variable(np.zeros([self.batch_size, self.params_dim]))
        self.reward_ph = tf.Variable(np.zeros([self.batch_size, 1]))
        # self.lr_ph = tf.placeholder(tf.float32, (), 'eta')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        # hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        # hid3_size = self.act_dim * 10  # 10 empirically determined
        # hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        # self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.input_ph, self.hidden_dim, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.params_dim)), name="h1")
        # out = tf.layers.dense(out, hid2_size, tf.tanh,
        #                       kernel_initializer=tf.random_normal_initializer(
        #                           stddev=np.sqrt(1 / hid1_size)), name="h2")
        # out = tf.layers.dense(out, hid3_size, tf.tanh,
        #                       kernel_initializer=tf.random_normal_initializer(
        #                           stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.rewards = tf.layers.dense(out, 1,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / self.hidden_dim)), name="rewards")
        self.rewards_sum = tf.reduce_sum(self.rewards)
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        # logvar_speed = (10 * hid3_size) // 48
        # log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
        #                            tf.constant_initializer(0.0))
        # self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        # print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
        #       .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))


    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        # if self.clipping_range is not None:
        #     print('setting up loss with clipping objective')
        #     pg_ratio = tf.exp(self.logp - self.logp_old)
        #     clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range[0], 1 + self.clipping_range[1])
        #     surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
        #                                 self.advantages_ph * clipped_pg_ratio)
        #     self.loss = -tf.reduce_mean(surrogate_loss)
        # else:
        #     print('setting up loss with KL penalty')
        #     loss1 = -tf.reduce_mean(self.advantages_ph *
        #                             tf.exp(self.logp - self.logp_old))
        #     loss2 = tf.reduce_mean(self.beta_ph * self.kl)
        #     loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        #     self.loss = loss1 + loss2 + loss3
        # optimizer = tf.train.AdamOptimizer(self.lr_ph)
        # self.train_op = optimizer.minimize(self.loss)
        self.loss = tf.reduce_mean(tf.pow(self.rewards - self.reward_ph, 2))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        optimizer_input = tf.train.GradientDescentOptimizer(self.lr)
        self.input_gradient_op = optimizer_input.compute_gradients(self.rewards_sum, var_list=[input_ph])

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)


    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
