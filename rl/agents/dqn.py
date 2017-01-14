import random
import os

import numpy as np
from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense

from agent import Agent
from ..scripts.experience_buffer import experience_buffer
from ..scripts.preprocess import get_state, get_positions

class QLearningAgent(Agent):
    """
    QLearning Agent based on https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

    # Arguments:
      mb_size: mini batch size
      state_size: agents observed state's size
      action_size: # of possible actions agent can make
      verbose: print to stdout
      epsilon: for epsilon probability do random action
      decay: lr decay
      min_epsilon: epsilon-greedy rate
      save_name: agent model name
      dataset_size: size of te replay memory
      number_of_episodes: number of episode
    """
    def __init__(self, mb_size=32, save_name='dqn', dataset_size=2000,\
        state_size=4, action_size=6,  \
        epsilon = 1.0, min_epsilon=0.1, decay=0.9, number_of_episodes=100000, \
        verbose = True):
        self.mb_size = mb_size
        self.save_name = save_name
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.verbose = verbose

        self.decay = decay
        self.min_epsilon = min_epsilon
        self.number_of_episodes = number_of_episodes
        self.dataset_size = dataset_size

        self.buffer = experience_buffer(buffer_size=self.dataset_size, reward_index=5)

        self.build_model()

    def build_model(self):
        """
        Builds neural networks. Loads previous weights autoamtically.
        """
        if self.save_name == 'dqn':
            model = Sequential()
            model.add(Dense(4, input_shape=(self.state_size+1,), activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(1, activation='linear'))
        elif self.save_name == 'dqn2':
            model = Sequential()
            model.add(Dense(8, input_shape=(self.state_size+1,), activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='linear'))

        model.compile(optimizer='rmsprop', loss='mse')
        if os.path.isfile(self.save_name + '.h5'):
            model.load_weights(self.save_name + '.h5')

        self.model = model

    def act(self, state):
        """
        Given a state, do an action using actor network

        Arguments:
         state : state of the environment
        """
        # argmax a Q(state, action)
        # 0,1 -> stay
        # 2,4 -> up
        # 3,5 -> down
        a_t = np.argmax(self.get_q_values(state))
        return a_t

    def get_q_values(self, state):
        """
        Returns Q values for all actions given a state.

        Arguments:
         state: state of the environment
        """
        q_values = np.zeros((self.action_size, ))
        for a in range(q_values.size):
            q_values[a] = self.model.predict(merge_state_action(state, a))
        return q_values

    def train(self, env):
        """
        Train agent in environment

        Arguments:
         env: gym environment
        """
        s_t = get_state(get_positions(env.reset()))

        for iepisode in range(self.number_of_episodes):
            if self.verbose:
                print 'Episode', iepisode

            for _ in range(3):
                replay_buffer = np.empty((self.dataset_size/2, self.state_size*2+2))

                for j in range(replay_buffer.shape[0]):
                    # select action a
                    if random.random() > self.epsilon:
                        a_t = self.act(s_t)
                        
                    else:
                        distance = s_t[0] - s_t[-1]
                        if distance > 0.2:
                            a_t = 2
                        elif distance > 0.08:
                            a_t = 4
                        elif distance < -0.2:
                            a_t = 3
                        elif distance < -0.08:
                            a_t = 5
                        else:
                            if random.random() < 0.5:
                                a_t = 0
                            else:
                                a_t = 1

                    if self.verbose:
                        env.render()

                    # take action and get reward
                    (o_t1, r_t, done, _) = env.step(a_t)

                    # get state
                    if  get_positions(o_t1)['distance'] <= 8.0:
                        r_t = 0.5
                    s_t1 = get_state(get_positions(o_t1))

                    # replay memory
                    replay_buffer[j, :] = package_replay(s_t, a_t, r_t, s_t1)

                    s_t = s_t1

                    # reset env
                    if done:
                        s_t = get_state(get_positions(env.reset()))

                self.buffer.add(replay_buffer)

            minibatch = self.buffer.sample(self.dataset_size)

            # create targets (no terminal state in pong)
            tts = np.zeros((minibatch.shape[0], 1))
            sss = np.zeros((minibatch.shape[0], s_t.size+1))
            for i in range(0, tts.shape[0]):
                (ss, aa, rr, ss_) = unpackage_replay(minibatch[i, :])
                qs = self.get_q_values(ss_)
                tts[i] = rr + self.decay * np.max(qs)
                sss[i, :] = merge_state_action(ss, aa)

            # train network
            self.model.fit(sss, tts, nb_epoch=2, batch_size=self.mb_size, verbose=False)
            self.model.save_weights(self.save_name + '.h5')

            # decay epsilon
            self.epsilon -= 1.0 / min(self.number_of_episodes, 1000)
            self.epsilon = max(self.min_epsilon, self.epsilon)
            if self.verbose:
                print 'Epsilon', self.epsilon

def merge_state_action(s_t, a_t):
    merged = np.zeros((1, s_t.size+1))
    merged[-1] = a_t
    return merged

def package_replay(s_t, a_t, r_t, s_t1):
    package = np.zeros((s_t.size*2 + 2))
    package[0:s_t.size] = s_t
    package[s_t.size] = a_t
    package[s_t.size+1] = r_t
    package[-s_t.size:] = s_t1
    return package[None, :]

def unpackage_replay(package):
    size = (package.size-2)/2
    s_t = package[:size]
    a_t = package[size]
    r_t = package[size+1]
    s_t1 = package[-size:]
    return (s_t, a_t, r_t, s_t1)
