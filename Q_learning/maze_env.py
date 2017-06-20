# -*- coding:utf-8 -*-
import numpy as np
np.random.seed(1)
import Tkinter as tk
import time

UNIT = 40
MAZE_H = 4
MAZE_W = 4

class Maze(tk.Tk, object):
	def __init__(self):
		super(Maze, self).__init__()
		self.action_space = ['u', 'd', 'l', 'r']
		self.n_actions = len(self.action_space)
		self.title('maze')
		self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
		self._build_maze()

	def _build_maze(self):
		self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

		# grids 만들기
		for c in range(0, MAZE_W * UNIT, UNIT):
			x0,y0,x1,y1 = c,0,c, MAZE_H * UNIT
			self.canvas.create_line(x0, y0, x1, y1)
		for r in range(0, MAZE_H * UNIT, UNIT):
			x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
			self.canvas.create_line(x0, y0, x1, y1)

		# origin 만들기
		origin = np.array([20,20])
		
		# hell
		hell1_center = origin + np.array([UNIT * 2, UNIT])
		self.hell1 = self.canvas.create_rectangle(
			hell1_center[0] - 15, hell1_center[1] - 15,
			hell1_center[0] + 15, hell1_center[1] + 15,
			fill = 'black')

		# hell
		hell2_center = origin + np.array([UNIT, UNIT * 2])
		self.hell2 = self.canvas.create_rectangle(
			hell2_center[0] - 15, hell2_center[1] - 15,
			hell2_center[0] + 15, hell2_center[1] + 15,
			fill='black')

		# oval 만들기
		oval_center = origin + UNIT * 2
		self.oval = self.canvas.create_oval(
			oval_center[0] - 15, oval_center[1] - 15,
			oval_center[0] + 15, oval_center[1] + 15,
			fill='yellow')

		# red rect 만들기
		self.rect = self.canvas.create_rectangle(
			origin[0] - 15, origin[1] - 15,
			origin[0] + 15, origin[1] + 15,
			fill='red')

		# observation 리턴
		return self.canvas.coords(self.rect)

	def step(self, action):
		s = self.canvas.coords(self.rect)
		base_action = np.array([0,0])
		if action == 0:
			if s[1] > UNIT:
				base_action[1] -= UNIT
		elif action == 1:
			if s[1] < (MAZE_H -1) * UNIT:
				base_action[1] += UNIT
		elif action == 2:
			if s[0] < (MAZE_W - 1) * UNIT:
				base_action[0] += UNIT
		elif action == 3:
			if s[0] > UNIT:
				base_action[0] -= UNIT
		
		# 무브 에이전트
		self.canvas.move(self.rect, base_action[0], base_action[1])

		s_ = self.canvas.coords(self.rect)

		# 보상 함수
		if s_ == self.canvas.coords(self.oval):
			reward = 1
			done = True
		elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
			reward = -1
			done = True
		else:
			reward = 0
			done = False

		return s_, reward, done

	def render(self):
		time.sleep(0.1)
		self.update()

