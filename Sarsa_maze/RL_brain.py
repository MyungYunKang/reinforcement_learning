# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

class RL(object):
	def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = action_space
		self.lr = learning_rate
		slef.gamma = reward_decay
		self.epsilon = e_greedy

		self.q_table = pd.DataFrame(columns=self.actions)
	def check_state_exist(self, state):
		if state not in self.q_table.index:
			self.q_table = self.q_table.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.q_table.columns,
					name=state,
					)
				)

	def choose_action(self, observation):
		self.check_state_exist(observation)

		if np.random.rand() < self.epsilon:
			# 최적의 행동을 선택
			state_action = self.q_table.ix[observation, :]
			state_action = state_action.reindex(np.random.permutation(state_action.index))
			action = state_action.argmax()
		else:
			action = np.random.choice(self.actions)
		return action

	def learn(self, *args):
		pass

class QLearningTable(RL):
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		q_predict = self.q_table.ix[s, a]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.ix[s_, :].max() # 다음 상태는 terminal이 아니다.
		else:
			q_target = r # 다음 상태는 terminal

		self.q_table.ix[s, a] += self.lr * (q_target - q_predict) # 업데이트

class SalsaTable(RL):
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		super(SalsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

	def learn(self, s, a, r, s_, a_):
		self.check_state_exist(s_)
		q_predict = self.q_table.ix[s, a]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.ix[s_, a_] # 다음 상태는 터미널이 아니다.
		else:
			q_target = r
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict) # 업데이트
