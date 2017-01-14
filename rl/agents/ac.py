import random, sys
import numpy as np

from agent import Agent
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD

from ..scripts.preprocess import get_state, get_positions

class ActorCriticAgent(Agent):
	"""
	Acton - Critic Agent based on https://github.com/gregretkowski/notebooks/blob/master/ActorCritic-with-OpenAI-Gym.ipynb

	# Arguments:
	  mb_size: mini batch size
	  state_size: agents observed state's size
      action_size: # of possible actions agent can make
	  verbose: print to stdout
	  actor_size: actor network hidden layer size
	  critic_size: critic network hidden layer size
	  epsilon: for epsilon probability do random action
      buffer_size: size of experience buffer
      epoch: # of times agent play game
      gamma: future discount coefficient
      lr: learning rate
      decay: lr decay
      momentum: lr momentum
      min_epsilon: epsilon-greedy rate
      save_name: agent model name
      load: load from a previously trained model
      frequency: save model every k epochs

	# References:

	TODO:
	 - render a game save to a file
	"""
	def __init__(self, mb_size = 32, state_size = None, action_size = None, verbose = True,
				 actor_size = 8, critic_size = 8, epsilon = 1.0, buffer_size = 4096,
				 epoch = 1000, gamma = 0.98, lr = 0.1, decay = 1e-6, momentum = 0.9, min_epsilon = 0.1, save_name = 'actor_critic', load = False, frequency = 5):

		self.mb_size = mb_size
		self.save_name = save_name
		self.state_size = state_size
		self.action_size = action_size
		self.verbose = verbose

		self.actor_size = actor_size
		self.critic_size = critic_size
		self.epsilon = epsilon
		self.buffer_size = buffer_size
		self.epoch = epoch
		self.gamma = gamma
		self.lr = lr
		self.decay = decay
		self.momentum = momentum
		self.min_epsilon = min_epsilon
		self.save_name = save_name
		self.frequency = frequency

		self.actor_replay = []
		self.critic_replay = []

		self.build_model(load)

	def build_model(self, load):
		"""
		Builds neural networks.

		Arguments:
		 load: whether load from previously trained model
		"""

		actor_model = Sequential()
		actor_model.add(Dense(self.actor_size, init='lecun_uniform', input_shape=(self.state_size,)))
		actor_model.add(Activation('relu'))
		actor_model.add(Dense(self.actor_size, init='lecun_uniform', input_shape=(self.state_size,)))
		actor_model.add(Activation('relu'))
		actor_model.add(Dense(self.actor_size, init='lecun_uniform', input_shape=(self.state_size,)))
		actor_model.add(Activation('relu'))
		actor_model.add(Dense(self.action_size, init='lecun_uniform'))
		actor_model.add(Activation('linear'))

		a_optimizer = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=True)
		actor_model.compile(loss='mse', optimizer=a_optimizer)

		critic_model = Sequential()

		critic_model.add(Dense(self.critic_size, init='lecun_uniform', input_shape=(self.state_size,)))
		critic_model.add(Activation('relu'))
		critic_model.add(Dense(self.critic_size, init='lecun_uniform', input_shape=(self.state_size,)))
		critic_model.add(Activation('relu'))
		critic_model.add(Dense(self.critic_size, init='lecun_uniform', input_shape=(self.state_size,)))
		critic_model.add(Activation('relu'))

		critic_model.add(Dense(1, init='lecun_uniform'))
		critic_model.add(Activation('linear'))

		c_optimizer = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=True)
		critic_model.compile(loss='mse', optimizer=c_optimizer)

		self.actor_model = actor_model
		self.critic_model= critic_model

		if load:
			print("loading a previous model")
			self.actor_model.load_weights('{}.actor.h5'.format(self.save_name))
			self.critic_model.load_weights('{}.critic.h5'.format(self.save_name))


	def act(self, state):
		"""
		Given a state, do an action using actor network

		Arguments:
		 state : state of the environment
		"""
		qval = self.actor_model.predict(state.reshape(1,self.state_size))
		action = (np.argmax(qval))
		return action

	def train(self, env):
		"""
		Train agent in environment

		Arguments:
         env: gym environment
		"""
		wins = 0
		losses = 0
		best_actor_cost = 0
		best_critic_cost = 0

		for i in range(self.epoch):
			observation = env.reset()
			done = False
			reward = 0
			info = None
			move_counter = 0
			won = False

			cost_actor = 0
			cost_critic = 0

			while(not done):

				orig_state = get_state(get_positions(observation))
				orig_reward = reward
				orig_val = self.critic_model.predict(orig_state.reshape(1,self.state_size))

				if (random.random() < self.epsilon): #choose random action
					action = np.random.randint(0,self.action_size)
				else: #choose best action from Q(s,a) values
					action = self.act(orig_state)

				#Take action, observe new state S'
				new_observation, new_reward, done, info = env.step(action)
				new_state = get_state(get_positions(new_observation))
				# Critic's value for this new state.
				new_val = self.critic_model.predict(new_state.reshape(1,self.state_size))

				if not done: # Non-terminal state.
					target = orig_reward + ( self.gamma * new_val)
				else:
					# In terminal states, the environment tells us the value directly.
					target = orig_reward + ( self.gamma * new_reward )

				best_val = max((orig_val * self.gamma), target)

				# Now append this to our critic replay buffer.
				self.critic_replay.append([orig_state,best_val])
				if done:
					self.critic_replay.append( [new_state, float(new_reward)])

				actor_delta = new_val - orig_val
				self.actor_replay.append([orig_state, action, actor_delta])

				if(len(self.critic_replay) >= self.buffer_size):
					minibatch = random.sample(self.critic_replay, self.mb_size)
					X_train = []
					y_train = []
					for memory in minibatch:
						m_state, m_value = memory
						y = np.empty([1])
						y[0] = m_value
						X_train.append(m_state.reshape((self.state_size,)))
						y_train.append(y.reshape((1,)))
					X_train = np.array(X_train)
					y_train = np.array(y_train)
					h = self.critic_model.fit(X_train, y_train, batch_size=self.mb_size, nb_epoch=1, verbose=0)
					cost_critic += h.history['loss'][-1]

				if(len(self.actor_replay) >= self.buffer_size):
					X_train = []
					y_train = []
					minibatch = random.sample(self.actor_replay, self.mb_size)
					for memory in minibatch:
						m_orig_state, m_action, m_value = memory
						old_qval = self.actor_model.predict( m_orig_state.reshape(1,self.state_size) )
						y = np.zeros(( 1, self.action_size ))
						y[:] = old_qval[:]
						y[0][m_action] = m_value
						X_train.append(m_orig_state.reshape((self.state_size,)))
						y_train.append(y.reshape((self.action_size,)))
					X_train = np.array(X_train)
					y_train = np.array(y_train)
					h = self.actor_model.fit(X_train, y_train, batch_size=self.mb_size, nb_epoch=1, verbose=0)
					cost_actor += h.history['loss'][-1]

				# Bookkeeping at the end of the turn.
				observation = new_observation
				reward = new_reward
				move_counter+=1
				if done:
					if new_reward > 0 : # Win
						wins += 1
						won = True
					else: # Loss
						losses += 1
					self.actor_replay = []
					self.actor_replay = []
#					self.actor_replay = random.sample(self.actor_replay, self.buffer_size / 10)
#					self.critic_replay = random.sample(self.critic_replay, self.buffer_size / 10)
			# Finised Epoch
			if self.verbose:
				print("Costs, actor %s  | critic %s" % (cost_actor, cost_critic))
				print("Game #: %s"% (i,))
				print("Moves this round %s" % move_counter)
				print("Wins/Losses %s/%s epsilon : %s" % (wins, losses, self.epsilon))
				sys.stdout.flush()
			### epsilon scheduling is totally ad-hoc
			if self.epsilon > self.min_epsilon:
				self.epsilon -= (1.0 / self.epoch)
			if won and self.epsilon > min_epsilon:
				self.epsilon -= 0.02
			if i % self.frequency == 0:
				print("saving the models....")
				self.actor_model.save_weights('{}.actor.h5'.format(self.save_name), True)
				self.critic_model.save_weights('{}.critic.h5'.format(self.save_name), True)
