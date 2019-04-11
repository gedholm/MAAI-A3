# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import gym

#Global parameters, Yolo. 
epochs = 2500;
learningRate = 0.001

batch_size = 10000
hidden_sizes = [5]
render = True
printBool = True

def mlp(x, sizes, activation=tf.nn.relu, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=15, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)


def makeAndtrainNetwork():
    #Parameters

    #enviroment 
    env = gym.make("BipedalWalker-v2")
    
    obs_dim =14
    n_acts = 6# env.action_space.n

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    reward_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    '''
    first layer tanh to catch -1 to 1 inputs
    '''
    FirstLayer= tf.layers.dense(obs_ph, units=15, activation=tf.tanh)
    '''
    feature extraction with mlp functions that build multiple layers 
    '''
    feature = mlp(FirstLayer, sizes=hidden_sizes)
    logits =tf.layers.dense(feature, units=15)
    
    '''
    4 diffrent "head" for the network, one for each action. 
    '''
    actionHead11 = tf.layers.dense(logits,units=n_acts, name='head11')
    actionHead12 = tf.layers.dense(actionHead11,units=n_acts, name='head12')
    actionHead13 = tf.layers.dense(actionHead12,units=n_acts, name='head13')
    actionHead1 = tf.layers.dense(actionHead13,units=n_acts, name='head1')
    
    actionHead21 = tf.layers.dense(logits,units=n_acts, name='head21')    
    actionHead22 = tf.layers.dense(actionHead21,units=n_acts, name='head22')
    actionHead23 = tf.layers.dense(actionHead22,units=n_acts, name='head23')
    actionHead2 = tf.layers.dense(actionHead23,units=n_acts, name='head2')
    
    actionHead31 = tf.layers.dense(logits,units=n_acts, name='head31')
    actionHead32 = tf.layers.dense(actionHead31,units=n_acts, name='head32')
    actionHead33 = tf.layers.dense(actionHead32,units=n_acts, name='head33')
    actionHead3 = tf.layers.dense(actionHead33,units=n_acts, name='head3')
     
    actionHead41 = tf.layers.dense(logits,units=n_acts, name='head41')
    actionHead42 = tf.layers.dense(actionHead41,units=n_acts, name='head42')
    actionHead43 = tf.layers.dense(actionHead42,units=n_acts, name='head43')
    actionHead4 = tf.layers.dense(actionHead43,units=n_acts, name='head4')
    
    #Baseline network - Tried to predict the reward
    B1 = tf.layers.dense(obs_ph,activation=tf.nn.relu,units=n_acts )
    B2 = tf.layers.dense(B1,activation=tf.nn.relu,units=n_acts )
    B3 = tf.layers.dense(B2,activation=tf.nn.relu,units=1 )
    
    baseLoss = tf.losses.mean_squared_error(tf.squeeze(B3),reward_ph)
    # make action selection op (outputs int actions, sampled from policy)
    #actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
    
    
    '''
    for training
    '''
    action1 = tf.squeeze(tf.multinomial(logits=(actionHead1),num_samples=1), axis=1)
    action2 = tf.squeeze(tf.multinomial(logits=(actionHead2),num_samples=1), axis=1)
    action3 = tf.squeeze(tf.multinomial(logits=(actionHead3),num_samples=1), axis=1)
    action4 = tf.squeeze(tf.multinomial(logits=( actionHead4),num_samples=1), axis=1)

    actionsum = tf.stack([action1,action2,action3,action4])

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    
    act_ph1 = tf.placeholder(shape=(None,), dtype=tf.int32)
    act_ph2 = tf.placeholder(shape=(None,), dtype=tf.int32)
    act_ph3 = tf.placeholder(shape=(None,), dtype=tf.int32)
    act_ph4 = tf.placeholder(shape=(None,), dtype=tf.int32)
    
    action_mask1 = tf.one_hot(act_ph1, n_acts)
    action_mask2 = tf.one_hot(act_ph2, n_acts)
    action_mask3 = tf.one_hot(act_ph3, n_acts)
    action_mask4 = tf.one_hot(act_ph4, n_acts)
    
    log_probs1 = tf.reduce_sum(action_mask1 * tf.nn.log_softmax(actionHead1), axis=1)
    log_probs2 = tf.reduce_sum(action_mask2 * tf.nn.log_softmax(actionHead2), axis=1)
    log_probs3 = tf.reduce_sum(action_mask3 * tf.nn.log_softmax(actionHead3), axis=1)
    log_probs4 = tf.reduce_sum(action_mask4 * tf.nn.log_softmax(actionHead4), axis=1)
    
    #loss1 = -tf.reduce_mean(tf.math.multiply( weights_ph , log_probs1))   
    loss1 =  -tf.reduce_mean(tf.math.multiply( weights_ph , log_probs1))
    loss2 =  -tf.reduce_mean(tf.math.multiply( weights_ph , log_probs2))
    loss3 = -tf.reduce_mean(tf.math.multiply( weights_ph , log_probs3))   
    loss4 =  -tf.reduce_mean(tf.math.multiply( weights_ph , log_probs4))
    
   
     
    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss1 +loss2+loss3+loss4)
    train_baseloss = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(baseLoss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    def train_one_epoch(prevRewardAvg, horizonLength):
        maxObs = np.zeros((16,))
        printBool = True
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_obsBuffer = []
        batch_acts1 = []         # for actions
        batch_acts2 = []         # for actions
        batch_acts3 = []         # for actions
        batch_acts4 = []         # for actions
        batch_acts1Buffer = []         # for actions
        batch_acts2Buffer = []         # for actions
        batch_acts3Buffer = []         # for actions
        batch_acts4Buffer = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rewards = []
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        totalReturn = []
        for j in range(1):
            # reset episode-specific variables
            obs = env.reset()       # first obs comes from starting distribution
            done = False            # signal from environment that episode is over
            ep_rews = []            # list for rewards accrued throughout ep
            
            # render first episode of each epoch
            finished_rendering_this_epoch = False
            iterMax =900
            iter = 0
            prevXSpeed = 0
            # collect experience by acting in the environment with current policy
            while True:
               
                obs = getClampedObservations(obs)[0:14].flatten()
                #maxObs = getMaxObs(maxObs,obs)
                #print(np.transpose(obs))
                # rendering
                if printBool:
                    #print(obs)
                    printBool = False
                if (not finished_rendering_this_epoch) and render and (np.mod(i,117) == 0) and j == 0:
                    env.render()
                    #print(obs)
                    #print(obs[2] , "x | y  " , obs[3])
    
                # save obs
                batch_obsBuffer.append(obs.copy())
    
                # act in the environment
                act = sess.run(actionsum, {obs_ph: obs.reshape(1,-1)})
                
                actionBucket= np.array([0.0,0.0,0.0,0.0])
                #print("this is Act",  act)
                for actionbuck in range(4):
                    #print(actionbuck)
                    actionValue = (float(act[actionbuck])/(n_acts-1)-0.5)*2
                    if actionValue < 0 :
                        actionValue = np.floor(actionValue)
                    else : 
                        actionValue = np.ceil(actionValue)
                    actionBucket[actionbuck] = 1* (float(act[actionbuck])/(n_acts-1)-0.5)*2
                #if (not finished_rendering_this_epoch) and render and (np.mod(i,3) == 0) and j == 0:
                #    print(actionBucket)  
                
                
                
                obs, rew, done, _ = env.step(actionBucket)
                #rew = rew+ 0.1*np.sum(actionBucket) 
                '''
                if obs[2] > 0:
                    rew +=obs[2]
                    prevXSpeed+=obs[2]
                else:
                    rew -=obs[2]
                    prevXSpeed = 0
                '''
                #print(actionBucket)
                # save action, reward
                batch_acts1Buffer.append(act[0])
                batch_acts2Buffer.append(act[1])
                batch_acts3Buffer.append(act[2])
                batch_acts4Buffer.append(act[3])
                ep_rews.append(rew)
                iter += 1
                if done or (iter > iterMax):
                    iter = 0
                    
                    # if episode is over, record info about episode
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    #print("reward",ep_ret)
                    if(ep_ret > -29000):
                        totalReturn.append(ep_ret)
                        batch_obs.extend( batch_obsBuffer.copy())
                        
                        batch_acts1.extend(batch_acts1Buffer.copy())
                        batch_acts2.extend(batch_acts2Buffer.copy())
                        batch_acts3.extend(batch_acts3Buffer.copy())
                        batch_acts4.extend(batch_acts4Buffer.copy())

                        batch_rets.append(ep_ret)
                        batch_lens.append(ep_len)
    
                        # the weight for each logprob(a|s) is R(tau)
                        #testing reward decay
                        #print("GOOD reward",ep_ret , "  ", len(batch_obs) ," / " , batch_size)
                        for r in range(ep_len):
                            cummulative = 0
                            for s in range(20):
                                cummulative += ep_rews[np.minimum(i+s,ep_len-1)]*np.power(0.6,s)
                            #baselineV = sess.run(B3, {obs_ph: obs.reshape(1,-1)})[0]
                            
                            if(cummulative > prevRewardAvg):
                                batch_weights += [cummulative]
                            else:
                                batch_weights += [cummulative]
                            batch_rewards += [cummulative]
                        #batch_weights+=[ ep_ret] * ep_len
        
                        # reset episode-specific variables
                    obs, done, ep_rews = env.reset(), False, []
                    batch_obsBuffer = []
                    batch_acts1Buffer = []
                    batch_acts2Buffer = []
                    batch_acts3Buffer = []
                    batch_acts4Buffer = []
                    
                    # won't render again this epoch
                    finished_rendering_this_epoch = True
    
                    # end experience loop if we have enough of it
                    if len(batch_obs) > batch_size:
                        break

            # take a single policy gradient update step
            ##print("act " ,np.size(np.array( batch_acts4)) )
            #print("wei " ,np.size(np.array(batch_weights)) )
            #print("obs " ,np.size(np.array(batch_obs)) )
            '''act_ph2: np.array(batch_acts2),
                                    act_ph3: np.array(batch_acts3),
                                    act_ph4: np.array(batch_acts4),
                                    weights_ph: np.array(batch_weights)
            '''
            #print(batch_weights)
            _ = sess.run(train_op,
                             feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    weights_ph: np.array(batch_weights).flatten(),
                                    act_ph1: np.array(batch_acts1).flatten(),
                                    act_ph2: np.array(batch_acts2).flatten(),
                                    act_ph3: np.array(batch_acts3).flatten(),
                                    act_ph4: np.array(batch_acts4).flatten()
                                    
                                 })
            
            _,bLoss = sess.run([train_baseloss,baseLoss] ,feed_dict = { obs_ph: np.array(batch_obs),
                                                      reward_ph: np.array(batch_rewards).flatten()})
        wAvg = np.mean(np.array(batch_rewards).flatten())
        #print(maxObs)
        return  batch_rets, batch_lens,bLoss, wAvg

    # training looprewardAvg 
    rewardAvg =0  
    horizonLength = 1
    for i in range(epochs):
        
        batch_rets, batch_lens,BL,rewardAvg = train_one_epoch(rewardAvg,horizonLength)
        horizonLength = int(np.floor(i/25))+1
                
        print('epoch: %3d %3f \t return: %.3f \t ep_len: %.3f '%
                (i,rewardAvg, np.mean(batch_rets), np.mean(batch_lens)))

  
    
def getMaxObs(oldMax, newObs):
    for i in range(len(newObs)):
        if(oldMax[i] <newObs[i]):
            oldMax[i]  = newObs[i]
    
    return oldMax
    
'''
Get rid of them pesky infitiy values 
'''
def getClampedObservations(observation):
    '''
    0	hull_angle	0	2*pi	0.5
    1	hull_angularVelocity	-inf	+inf	-
    2	vel_x	-1	+1	-
    3	vel_y	-1	+1	-
    4	hip_joint_1_angle	-inf	+inf	-
    5	hip_joint_1_speed	-inf	+inf	-
    6	knee_joint_1_angle	-inf	+inf	-
    7	knee_joint_1_speed	-inf	+inf	-
    8	leg_1_ground_contact_flag	0	1	-
    9	hip_joint_2_angle	-inf	+inf	-
    10	hip_joint_2_speed	-inf	+inf	-
    11	knee_joint_2_angle	-inf	+inf	-
    12	knee_joint_2_speed	-inf	+inf	-
    13	leg_2_ground_contact_flag	0	1	-
    14-23	10 lidar readings	-inf	+inf	-
    '''
    #get everything in the intervall [0,1]
    maxJointAngle = 2*np.pi;
    maxLidarDistance = 10;
    maxJointSpeed = 2;
    #print(np.size(observation))
    clampedObservation = np.zeros(np.size(observation))
    
    
    clampedObservation[0] = observation[0]/(2*np.pi)
    clampedObservation[1] = np.minimum(np.maximum(observation[1]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[2] = observation[2]*0.5
    clampedObservation[3] = observation[3]*0.5
    clampedObservation[4] = (observation[4]-np.floor(observation[4]/(2*np.pi))*(2*np.pi))/(maxJointAngle)
    clampedObservation[5] = np.minimum(np.maximum(observation[5]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[6] = (observation[6]-np.floor(observation[6]/(2*np.pi))*(2*np.pi))/(maxJointAngle)
    clampedObservation[7] = np.minimum(np.maximum(observation[7]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[8] = observation[8]
    clampedObservation[9] = (observation[9]-np.floor(observation[9]/(2*np.pi))*(2*np.pi))/(maxJointAngle)
    clampedObservation[10] = np.minimum(np.maximum(observation[10]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[11] = (observation[11]-np.floor(observation[11]/(2*np.pi))*(2*np.pi))/(maxJointAngle)
    clampedObservation[12] = np.minimum(np.maximum(observation[12]/(2*maxJointSpeed) + 0.5,0),1)
    clampedObservation[13] = observation[13]

    for o in range(14,23):
        clampedObservation[o] = np.minimum(np.maximum(observation[o]/(2*maxLidarDistance) + 0.5,0),1)
    
    return clampedObservation
    
if __name__ == '__main__':
    makeAndtrainNetwork()