"""
localize an object in an image
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class ObjectLocalizationEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self):

		high = 255
		low = 0

		self.action_space = spaces.Discrete(4)  # up right left down
		self.observation_space = spaces.Box(low, high, (600,400,3))

		self._seed()
		self.screen_width = 600
		self.screen_heigth = 400

		self.frame_width = 40
		self.frame_heigth = 40

		self.target_width = 40 
		self.target_heigth = 40

		self.target_l = self.np_random.random_integers(low = self.frame_width , high = self.screen_width - self.target_width)
		self.target_t = self.np_random.random_integers(low = self.frame_heigth , high = self.screen_heigth - self.target_heigth)

		self.viewer = None

	def _iou(self,boxA, boxB): # is frame on the right object?
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])

		interArea = (xB - xA + 1) * (yB - yA + 1)

		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

		iou = interArea / float(boxAArea + boxBArea - interArea)

		return iou

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state

		position, image = self.state
		frame_x, frame_y  = position

		delta = 2
		if action == 0:      # perform action
			frame_x += delta
		elif action == 1:
			frame_y += delta
		elif action == 2:
			frame_x -= delta
		else:
			frame_y -= delta

		new_image = self._render(mode = 'rgb_array') # move frame according to the action
		self.state = (frame_x , frame_y), new_image

		pos_x = self.screen_width/2 + frame_x
		pos_y = self.screen_heigth/2 + frame_y

		threshold = 0.05
		intersect = self._iou([pos_x + self.frame_width, pos_y, pos_x, pos_y - self.frame_heigth], [self.target_l + self.target_width, self.target_t, self.target_l, self.target_t - self.target_heigth])
		if intersect >= threshold or pos_x < 0 or pos_x > self.screen_width or pos_y < 0 or pos_y > self.screen_heigth: # is frame on the target or out of screen?
			done = True
		else:
			done = False

		if done:
			if intersect >= threshold:
				reward = intersect
			else:
				reward = -10
		else:
			reward = 0
		return new_image, reward, done, {}

	def _reset(self):

		position = (0, 0)

		self.target_l = self.np_random.random_integers(low = self.frame_width , high = self.screen_width - self.target_width)
		self.target_t = self.np_random.random_integers(low = self.frame_heigth , high = self.screen_heigth - self.target_heigth)

		self.fake_list = []  # place fake objects
		for i in xrange(40):
			self.fake_list += [self.np_random.random_integers(low = self.frame_width , high = self.screen_width - self.target_width, size = (2,))]

		self.state = position, np.zeros((600,400,3))
		image = self._render(mode = 'rgb_array')
		self.state = position, image

		return self.state

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		if self.viewer is None:  
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(self.screen_width, self.screen_heigth)

			fakes = []
			for pair in self.fake_list: # draw fake objects
				l,t = pair
				fakes += [rendering.FilledPolygon([(l, t + self.target_heigth ), (l, t), (l + self.target_width, t), (l + self.target_width, t + self.target_heigth)])]
				fakes[-1].set_color(.9,.9,.9)
				self.viewer.add_geom(fakes[-1])

			# draw target object
			target = rendering.FilledPolygon([(self.target_l, self.target_t - self.target_heigth ), (self.target_l, self.target_t), (self.target_l + self.target_width, self.target_t), (self.target_l + self.target_width, self.target_t - self.target_heigth)])
			target.set_color(.5,.5,.8)
			self.viewer.add_geom(target)

			# draw frame
			frame = rendering.PolyLine([( self.screen_width/2 , self.screen_heigth/2 - self.frame_heigth), ( self.screen_width/2, self.screen_heigth/2 ),( self.screen_width/2 + self.frame_width, self.screen_heigth/2),( self.screen_width/2 + self.frame_width, self.screen_heigth/2 - self.frame_heigth)], True)
			frame.set_color(.8,.6,.4)
			self.frametrans = rendering.Transform()
			frame.add_attr(self.frametrans)
			self.viewer.add_geom(frame)


		# move the frame
		x, _ = self.state
		self.frametrans.set_translation(x[0], x[1])
		return self.viewer.render(return_rgb_array = mode=='rgb_array')
