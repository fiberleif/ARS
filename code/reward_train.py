from reward_func import *
import numpy as np
from random import shuffle 

w_policy_size = 102
hidden_dim = 100
batch_size = 256
seed = 0
lr = 1e-2

dataset_x = np.loadtxt("dataset_x.txt")
dataset_y = np.loadtxt("dataset_y.txt")
# print(dataset_y.shape)
# for i in range(dataset_y.shape[0]):
#     dataset_y[i] = np.sign(dataset_y[i]) * np.sqrt(abs(dataset_y[i]))

dataset_y = dataset_y.reshape(-1,1)

reward_func = RewardFunction(params_dim= w_policy_size, hidden_dim= hidden_dim, lr= lr, seed= seed)

train_size = dataset_x.shape[0]
# print(train_size)
num_batch = int(train_size / batch_size)
reward_train_epochs = 50000
# output reward loss before train
loss_list = []
# loss_list_var = []
for i in range(num_batch):
    loss_list.append(reward_func.sess.run(reward_func.loss, feed_dict={reward_func.input_ph: dataset_x[i*batch_size: (i+1)*batch_size], 
                                reward_func.reward_ph: dataset_y[i*batch_size: (i+1)*batch_size]}))
    # loss_list_var.append(reward_func.sess.run(reward_func.loss, feed_dict={reward_func.input_var: dataset_x[i*batch_size: (i+1)*batch_size], 
    #                             reward_func.reward_var: dataset_y[i*batch_size: (i+1)*batch_size]}))
# print(loss_list)
reward_avg_loss = np.mean(loss_list)
# reward_avg_loss_var = np.mean(loss_list_var)
print("reward_avg_loss_before:", reward_avg_loss)
# print("reward_avg_loss_var_before:", reward_avg_loss_var)



# train reward function
for i in range(reward_train_epochs):
    init_list = [i for i in range(train_size)]
    shuffle(init_list)
    shuffle_dataset_x = dataset_x[init_list]
    shuffle_dataset_y = dataset_y[init_list]
    reward_func.sess.run(reward_func.train_op, feed_dict={reward_func.input_ph: shuffle_dataset_x[0:batch_size], 
                            reward_func.reward_ph: shuffle_dataset_y[0:batch_size]})
    if(i%200 == 0) and (i != 0):
        # output reward function loss
        loss_list = []
        # loss_list_var = []
        for j in range(num_batch):
            loss_list.append(reward_func.sess.run(reward_func.loss, feed_dict={reward_func.input_ph: dataset_x[j*batch_size: (j+1)*batch_size], 
                                reward_func.reward_ph: dataset_y[j*batch_size: (j+1)*batch_size]}))
            # loss_list_var.append(reward_func.sess.run(reward_func.loss, feed_dict={reward_func.input_var: dataset_x[j*batch_size: (j+1)*batch_size], 
            #                     reward_func.reward_var: dataset_y[j*batch_size: (j+1)*batch_size]}))
        reward_avg_loss = np.mean(loss_list)
        # reward_avg_loss_var = np.mean(loss_list_var)
        print("epoch {0}: reward_avg_loss:{1}".format(i, reward_avg_loss))
        # print("epoch {0}: reward_avg_loss_var:{1}".format(i, reward_avg_loss_var))


