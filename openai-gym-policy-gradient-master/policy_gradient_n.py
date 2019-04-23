import tensorflow as tf
import numpy as np
import random
from tensorflow.python.framework import ops
from functions_actions_N import *

class PolicyGradientN:
	def __init__(
        self,
        n_x,
        n_y = 3,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None,
        N=4,
        iteration=0):
		self.n_x = n_x
		self.n_y = n_y
		self.lr = learning_rate
		self.gamma = reward_decay
		self.N = N
		self.iteration = iteration
		self.save_path = None
		if save_path is not None:
		    self.save_path = save_path

		self.reset_episode_data()
		self.build_network()

		self.cost_history = []

		self.sess = tf.Session()

		# $ tensorboard --logdir=logs
		# http://0.0.0.0:6006/
		tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

		# 'Saver' op to save and restore all the variables
		self.saver = tf.train.Saver()

		# Restore model
		if load_path is not None:
		    self.load_path = load_path
		    self.saver.restore(self.sess, self.load_path)


	def store_transition(self, s, a1, a2, a3, a4, r):
		"""
		    Store play memory for training

		    Arguments:
		        s: observation
		        a: action taken
		        r: reward after action
		"""
		self.episode_observations.append(s)
		self.episode_rewards.append(r)
		# Store actions as list of arrays
		# e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
		# action = np.zeros(self.n_y)
		# action[a] = 1

		self.episode_actions[0].append(a1)
		self.episode_actions[1].append(a2)
		self.episode_actions[2].append(a3)
		self.episode_actions[3].append(a4)

	def choose_action(self, observation):
		"""
		    Choose action based on observation

		    Arguments:
		        observation: array of state, has shape (num_features)

		    Returns: index of action we want to choose
		"""
		# Reshape observation to (num_features, 1)
		observation = observation[:, np.newaxis]

		# Run forward propagation to get softmax probabilities
		prob_weights_1 = self.sess.run(self.outputs_softmax_1, feed_dict = {self.X: observation})
		prob_weights_2 = self.sess.run(self.outputs_softmax_2, feed_dict = {self.X: observation})
		prob_weights_3 = self.sess.run(self.outputs_softmax_3, feed_dict = {self.X: observation})
		prob_weights_4 = self.sess.run(self.outputs_softmax_4, feed_dict = {self.X: observation})

		prob_raveled_1 = prob_weights_1.ravel()
		prob_raveled_2 = prob_weights_2.ravel()
		prob_raveled_3 = prob_weights_3.ravel()
		prob_raveled_4 = prob_weights_4.ravel()
		# Select action using a biased sample
		# this will return the index of the action we've sampled
		a_1 = np.random.choice(range(len(prob_raveled_1)), p=prob_raveled_1)
		a_2 = np.random.choice(range(len(prob_raveled_2)), p=prob_raveled_2)
		a_3 = np.random.choice(range(len(prob_raveled_3)), p=prob_raveled_3)
		a_4 = np.random.choice(range(len(prob_raveled_4)), p=prob_raveled_4)

		# a_1 = np.argmax(prob_raveled_1)
		# a_2 = np.argmax(prob_raveled_2)
		# a_3 = np.argmax(prob_raveled_3)
		# a_4 = np.argmax(prob_raveled_4)

		one_hot_1 = make_one_hot_int(a_1)
		one_hot_2 = make_one_hot_int(a_2)
		one_hot_3 = make_one_hot_int(a_3)
		one_hot_4 = make_one_hot_int(a_4)


		return one_hot_1, one_hot_2, one_hot_3, one_hot_4

	def reset_episode_data(self):
		self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []
		for i in range(0, self.N):
			self.episode_actions.append([])

	def learn(self):
		# Discount and normalize episode reward
		if len(self.episode_observations) == 0:
			self.episode_observations.append([0]*self.n_x)
			fake_action = [0, 0, 0, 0]
			oh1, oh2, oh3, oh4 = make_one_hots_action(fake_action)
			self.episode_actions[0].append(oh1)
			self.episode_actions[1].append(oh2)
			self.episode_actions[2].append(oh3)
			self.episode_actions[3].append(oh4)

			self.episode_rewards.append(0.0)
		discounted_episode_rewards_norm = self.discount_and_norm_rewards()

		# Train on episode
		self.sess.run(self.train_op_1, feed_dict={
		     self.X: np.vstack(self.episode_observations).T,
		     self.Y: np.vstack(np.array(self.episode_actions[0])).T,
		     self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
		})

		self.sess.run(self.train_op_2, feed_dict={
		     self.X: np.vstack(self.episode_observations).T,
		     self.Y: np.vstack(np.array(self.episode_actions[1])).T,
		     self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
		})

		self.sess.run(self.train_op_3, feed_dict={
		     self.X: np.vstack(self.episode_observations).T,
		     self.Y: np.vstack(np.array(self.episode_actions[2])).T,
		     self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
		})

		self.sess.run(self.train_op_4, feed_dict={
		     self.X: np.vstack(self.episode_observations).T,
		     self.Y: np.vstack(np.array(self.episode_actions[3])).T,
		     self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
		})

		# Reset the episode data
		self.reset_episode_data()
		#self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

		# Save checkpoint
		if self.save_path is not None:
		    save_path = self.saver.save(self.sess, self.save_path)
		    print("Model saved in file: %s" % save_path)

		return discounted_episode_rewards_norm

	def discount_and_norm_rewards(self):
		discounted_episode_rewards = np.zeros_like(self.episode_rewards)
		cumulative = 0
		for t in reversed(range(len(self.episode_rewards))):
		    cumulative = cumulative * self.gamma + self.episode_rewards[t]
		    discounted_episode_rewards[t] = cumulative

		discounted_episode_rewards -= np.mean(discounted_episode_rewards)
		discounted_episode_rewards /= (np.std(discounted_episode_rewards) + 1*10**(-9))
		return discounted_episode_rewards

	def build_network(self):
		# Create placeholders
		seeds = np.array([2, 3, 4, 1])
		seeds += self.iteration

		#random.shuffle(seeds)
		with tf.name_scope('inputs'):
		    self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
		    self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
		    self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

		# Initialize parameters
		units_layer_1 = 10
		units_layer_2 = 7
		units_output_layer = self.n_y
		with tf.name_scope('parameters'):
		    W1_1 = tf.get_variable("W1_1"+str(self.iteration), [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[1]))
		    b1_1 = tf.get_variable("b1_1"+str(self.iteration), [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[1]))
		    W2_1 = tf.get_variable("W2_1"+str(self.iteration), [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[1]))
		    b2_1 = tf.get_variable("b2_1"+str(self.iteration), [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[1]))
		    W3_1 = tf.get_variable("W3_1"+str(self.iteration), [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[1]))
		    b3_1 = tf.get_variable("b3_1"+str(self.iteration), [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[1]))

		    W1_2 = tf.get_variable("W1_2"+str(self.iteration), [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[2]))
		    b1_2 = tf.get_variable("b1_2"+str(self.iteration), [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[2]))
		    W2_2 = tf.get_variable("W2_2"+str(self.iteration), [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[2]))
		    b2_2 = tf.get_variable("b2_2"+str(self.iteration), [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[2]))
		    W3_2 = tf.get_variable("W3_2"+str(self.iteration), [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[2]))
		    b3_2 = tf.get_variable("b3_2"+str(self.iteration), [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[2]))

		    W1_3 = tf.get_variable("W1_3"+str(self.iteration), [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[3]))
		    b1_3 = tf.get_variable("b1_3"+str(self.iteration), [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[3]))
		    W2_3 = tf.get_variable("W2_3"+str(self.iteration), [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[3]))
		    b2_3 = tf.get_variable("b2_3"+str(self.iteration), [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[3]))
		    W3_3 = tf.get_variable("W3_3"+str(self.iteration), [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[3]))
		    b3_3 = tf.get_variable("b3_3"+str(self.iteration), [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[3]))

		    W1_4 = tf.get_variable("W1_4"+str(self.iteration), [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[0]))
		    b1_4 = tf.get_variable("b1_4"+str(self.iteration), [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[0]))
		    W2_4 = tf.get_variable("W2_4"+str(self.iteration), [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[0]))
		    b2_4 = tf.get_variable("b2_4"+str(self.iteration), [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[0]))
		    W3_4 = tf.get_variable("W3_4"+str(self.iteration), [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[0]))
		    b3_4 = tf.get_variable("b3_4"+str(self.iteration), [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=seeds[0]))
		# Forward prop
		with tf.name_scope('layer_1'):
		    Z1_1 = tf.add(tf.matmul(W1_1,self.X), b1_1)
		    A1_1 = tf.nn.relu(Z1_1)

		    Z1_2 = tf.add(tf.matmul(W1_2,self.X), b1_2)
		    A1_2 = tf.nn.relu(Z1_2)

		    Z1_3 = tf.add(tf.matmul(W1_3,self.X), b1_3)
		    A1_3 = tf.nn.relu(Z1_3)

		    Z1_4 = tf.add(tf.matmul(W1_4,self.X), b1_4)
		    A1_4 = tf.nn.relu(Z1_4)
		with tf.name_scope('layer_2'):
		    Z2_1 = tf.add(tf.matmul(W2_1, A1_1), b2_1)
		    A2_1 = tf.nn.relu(Z2_1)

		    Z2_2 = tf.add(tf.matmul(W2_2, A1_2), b2_2)
		    A2_2 = tf.nn.relu(Z2_2)

		    Z2_3 = tf.add(tf.matmul(W2_3, A1_3), b2_3)
		    A2_3 = tf.nn.relu(Z2_3)

		    Z2_4 = tf.add(tf.matmul(W2_4, A1_4), b2_4)
		    A2_4 = tf.nn.relu(Z2_4)
		with tf.name_scope('layer_3'):
		    Z3_1 = tf.add(tf.matmul(W3_1, A2_1), b3_1)
		    A3_1 = tf.nn.softmax(Z3_1)

		    Z3_2 = tf.add(tf.matmul(W3_2, A2_2), b3_2)
		    A3_2 = tf.nn.softmax(Z3_2)

		    Z3_3 = tf.add(tf.matmul(W3_3, A2_3), b3_3)
		    A3_3 = tf.nn.softmax(Z3_3)

		    Z3_4 = tf.add(tf.matmul(W3_4, A2_4), b3_4)
		    A3_4 = tf.nn.softmax(Z3_4)

		# Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
		logits_1 = tf.transpose(Z3_1)
		labels_1 = tf.transpose(self.Y)
		self.outputs_softmax_1 = tf.nn.softmax(logits_1, name='A3_1')

		logits_2 = tf.transpose(Z3_2)
		labels_2 = tf.transpose(self.Y)
		self.outputs_softmax_2 = tf.nn.softmax(logits_2, name='A3_2')

		logits_3 = tf.transpose(Z3_3)
		labels_3 = tf.transpose(self.Y)
		self.outputs_softmax_3 = tf.nn.softmax(logits_3, name='A3_3')

		logits_4 = tf.transpose(Z3_4)
		labels_4 = tf.transpose(self.Y)
		self.outputs_softmax_4 = tf.nn.softmax(logits_4, name='A3_4')

		with tf.name_scope('loss'):
		    neg_log_prob_1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_1, labels=labels_1)
		    loss_1 = tf.reduce_mean(neg_log_prob_1 * self.discounted_episode_rewards_norm)  # reward guided loss

		    neg_log_prob_2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=labels_2)
		    loss_2 = tf.reduce_mean(neg_log_prob_2 * self.discounted_episode_rewards_norm)  # reward guided loss

		    neg_log_prob_3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_3, labels=labels_3)
		    loss_3 = tf.reduce_mean(neg_log_prob_3 * self.discounted_episode_rewards_norm)  # reward guided loss

		    neg_log_prob_4 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_4, labels=labels_4)
		    loss_4 = tf.reduce_mean(neg_log_prob_4 * self.discounted_episode_rewards_norm)  # reward guided loss

		with tf.name_scope('train'):
		    self.train_op_1 = tf.train.AdamOptimizer(self.lr).minimize(loss_1)

		    self.train_op_2 = tf.train.AdamOptimizer(self.lr).minimize(loss_2)

		    self.train_op_3 = tf.train.AdamOptimizer(self.lr).minimize(loss_3)

		    self.train_op_4 = tf.train.AdamOptimizer(self.lr).minimize(loss_4)

	def run_simulation(self, max_frames, env, render):
		observation = env.reset()[0:14]
		frame_counter = 0
		done = False
		while not done:
		    frame_counter += 1
		    if render: env.render()
		    one_hot_1, one_hot_2, one_hot_3, one_hot_4 = self.choose_action(observation)
		    a1 = make_action_from_one(one_hot_1, 0)
		    a2 = make_action_from_one(one_hot_2, 1)
		    a3 = make_action_from_one(one_hot_3, 2)
		    a4 = make_action_from_one(one_hot_4, 3)
		    action = sum_actions(a1, a2, a3, a4)
		    observation, reward, done, info = env.step(action)
		    observation = observation[0:14]
		    self.store_transition(observation, one_hot_1, one_hot_2, one_hot_3, one_hot_4, reward)
		    if frame_counter > max_frames: done = True
		    if done:
		        rewards_sum = sum(self.episode_rewards)
		        self.reset_episode_data()
		        print("Simulation reward: ", rewards_sum)
		        return(rewards_sum)
