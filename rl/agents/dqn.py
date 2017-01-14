import random

import numpy as np
from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense

from agent import Agent
from ..scripts.experience_buffer import experience_buffer
from ..scripts.preprocess import get_state, get_positions

class QLearningAgent(Agent):

    def __init__(self, mb_size=100, save_name='dqn', state_size=4, action_size=6,  \
        epsilon = 1.0, min_epsilon=0.1, decay=0.9, number_of_episodes=1000, \
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

        self.buffer = experience_buffer(buffer_size=1000, reward_index=5)

        self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(4, input_shape=(self.state_size+1,), activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='rmsprop', loss='mse')

        self.model = model

    def act(self, state):
        # argmax a Q(state, action)
        # 0,1 -> stay
        # 2,4 -> up
        # 3,5 -> down
        a_t = np.argmax(self.get_q_values(state))
        if self.verbose:
            print 'argmax:', a_t
        return a_t

    def get_q_values(self, state):
        q_values = np.zeros((self.action_size, ))
        for a in range(q_values.size):
            q_values[a] = self.model.predict(merge_state_action(state, a))
        return q_values

    def train(self, env):
        s_t = get_state(get_positions(env.reset()))

        for iepisode in range(self.number_of_episodes):
            if self.verbose:
                print 'Episode', iepisode

            for _ in range(3):
                replay_buffer = np.empty((self.mb_size, self.state_size*2+2))

                for j in range(self.mb_size):# stack states; read about q learning
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
                    print get_positions(o_t1)['distance']
                    s_t1 = get_state(get_positions(o_t1))

                    # replay memory
                    replay_buffer[j, :] = package_replay(s_t, a_t, r_t, s_t1)

                    s_t = s_t1

                    # reset env
                    if done:
                        s_t = get_state(get_positions(env.reset()))

                self.buffer.add(replay_buffer)

            minibatch = self.buffer.sample(self.mb_size)

            # create targets (no terminal state in pong)
            tts = np.zeros((minibatch.shape[0], 1))
            sss = np.zeros((minibatch.shape[0], s_t.size+1))
            for i in range(0, tts.shape[0]):
                (ss, aa, rr, ss_) = unpackage_replay(minibatch[i, :])
                qs = self.get_q_values(ss_)
                tts[i] = rr + self.decay * np.max(qs)
                sss[i, :] = merge_state_action(ss, aa)

            # train network
            self.model.fit(sss, tts, nb_epoch=100, batch_size=20)

            # decay epsilon
            self.epsilon -= 1 / self.number_of_episodes
            self.epsilon = min(self.min_epsilon, self.epsilon)

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
