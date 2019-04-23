# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import gym

#Global parameters, Yolo. 
epochs = 2500;
learningRate = 0.00001
maxLidarDistance = 10
maxJointSpeed = 1
batch_size = 1000
hidden_sizes = [20, 10]
render = True

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)


def makeAndtrainNetwork():
    #Parameters

    #enviroment 
    env = gym.make("BipedalWalker-v2")
    
    obs_dim = env.observation_space.shape[0]
    n_acts = 9# env.action_space.n

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)
    actionBucket= np.array([0,0,0,0])
    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)

    negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=act_ph, logits=actions)
    weighted_negative_likelihoods = tf.multiply(negative_likelihoods, weights_ph)
    loss = tf.reduce_mean(weighted_negative_likelihoods)
    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        for j in range(1):
            # reset episode-specific variables
            obs = env.reset()       # first obs comes from starting distribution
            done = False            # signal from environment that episode is over
            ep_rews = []            # list for rewards accrued throughout ep
    
            # render first episode of each epoch
            finished_rendering_this_epoch = False
            iterMax = 500
            iter = 0
            # collect experience by acting in the environment with current policy
            while True:
                obs = getClampedObservations(obs)
                # rendering
                
                if (not finished_rendering_this_epoch) and render and (np.mod(i,20) == 0) and j == 0:
                    env.render()
    
                # save obs
                batch_obs.append(obs.copy())
    
                # act in the environment
                if (False):
                    act = env.heuristcWalk(obs, done)
                    obs, rew, done, _ = env.step(act)
                else:
                    act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
                    actionBucket= np.array([0,0,0,0])
                    
                    if( not act>=8):
                        actionBucket[int(np.floor(act/2))] = (act -2*np.floor(act/2)-0.5)*2
                    obs, rew, done, _ = env.step(actionBucket)
                #print(actionBucket)
                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)
                iter += 1
                if done or (iter > iterMax):
                    iter = 0
                    # if episode is over, record info about episode
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)
    
                    # the weight for each logprob(a|s) is R(tau)
                    #testing reward decay
                    for r in range(ep_len):
                        cummulative = 0
                        for s in range(20):
                            cummulative += ep_rews[np.minimum(i+s,ep_len-1)]*np.power(0.99,s)
                        batch_weights += [cummulative]
                    #batch_weights+=[ ep_ret] * ep_len
    
                    # reset episode-specific variables
                    obs, done, ep_rews = env.reset(), False, []
    
                    # won't render again this epoch
                    finished_rendering_this_epoch = True
    
                    # end experience loop if we have enough of it
                    if len(batch_obs) > batch_size:
                        break

            # take a single policy gradient update step
            #print("act " ,np.size(np.array( batch_acts)) )
            #print("wei " ,np.size(np.array(batch_weights)) )
            batch_loss, _ = sess.run([loss, train_op],
                             feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

  
    
    
    
    
    
'''
Get rid of them pesky infitiy values 
'''
def getClampedObservations(observation):
    '''
    0   hull_angle  0   2*pi    0.5
    1   hull_angularVelocity    -inf    +inf    -
    2   vel_x   -1  +1  -
    3   vel_y   -1  +1  -
    4   hip_joint_1_angle   -inf    +inf    -
    5   hip_joint_1_speed   -inf    +inf    -
    6   knee_joint_1_angle  -inf    +inf    -
    7   knee_joint_1_speed  -inf    +inf    -
    8   leg_1_ground_contact_flag   0   1   -
    9   hip_joint_2_angle   -inf    +inf    -
    10  hip_joint_2_speed   -inf    +inf    -
    11  knee_joint_2_angle  -inf    +inf    -
    12  knee_joint_2_speed  -inf    +inf    -
    13  leg_2_ground_contact_flag   0   1   -
    14-23   10 lidar readings   -inf    +inf    -
    '''
    #get everything in the intervall [0,1]
    jointAngleModulus = 2*np.pi;
    maxLidarDistance = 10;
    maxJointSpeed = 1;
    #print(np.size(observation))
    clampedObservation = np.zeros(np.size(observation))
    
    
    clampedObservation[0] = observation[0]/(2*np.pi)
    clampedObservation[1] = np.minimum(np.maximum(observation[1]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[2] = observation[2]*0.5
    clampedObservation[3] = observation[3]*0.5
    clampedObservation[4] = (observation[4]-np.floor(observation[4]/(2*np.pi))*(2*np.pi))/(2*np.pi)
    clampedObservation[5] = np.minimum(np.maximum(observation[5]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[6] = (observation[6]-np.floor(observation[6]/(2*np.pi))*(2*np.pi))/(2*np.pi)
    clampedObservation[7] = np.minimum(np.maximum(observation[7]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[8] = observation[8]
    clampedObservation[9] = (observation[9]-np.floor(observation[9]/(2*np.pi))*(2*np.pi))/(2*np.pi)
    clampedObservation[10] = np.minimum(np.maximum(observation[10]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[11] = (observation[11]-np.floor(observation[11]/(2*np.pi))*(2*np.pi))/(2*np.pi)
    clampedObservation[12] = np.minimum(np.maximum(observation[12]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[13] = observation[13]

    for o in range(14,23):
        clampedObservation[o] = np.minimum(np.maximum(observation[o]/(2*maxLidarDistance) + 0.5,0),1)
    
    return clampedObservation
    
if __name__ == '__main__':
    makeAndtrainNetwork()