# -*- coding:utf-8 -*-
"""
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
	def __init__(
			self, n_actions, n_features,
			learning_rate = 0.01
			reward_decay = 0.9
			e_greedy = 0.9
			replace_target_iter=300,
			memory_size=500,
			batch_size=32,
			e_greedy_increment=None,
			output_graph=False,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self
