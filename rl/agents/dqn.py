import random
import os

import numpy as np

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
    def __init__(self, mb_size=64, save_name='dqn_temporal', dataset_size=10000,\
        state_size=6, action_size=3, verbose=False, temporal_chain=2,\
        epsilon=1.0, min_epsilon=0.1, decay=0.99, number_of_episodes=100000):
        self.mb_size = mb_size
        self.save_name = save_name
        self.state_size = 6 #state_size
        self.action_size = 3 # action_size
        self.epsilon = epsilon
        self.verbose = verbose

        self.decay = decay
        self.min_epsilon = min_epsilon
        self.number_of_episodes = number_of_episodes
        self.dataset_size = dataset_size
        self.temporal_chain = temporal_chain

        self.buffer = experience_buffer(buffer_size=20*self.dataset_size, \
            reward_index=1, action_index=0)

        self.build_model()

    def build_model(self):
        """
        Builds neural networks. Loads previous weights autoamtically.
        """
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size*self.temporal_chain+3,), activation='relu', init='lecun_uniform'))
        model.add(Dense(24, activation='relu', init='lecun_uniform'))
        model.add(Dense(16, activation='relu', init='lecun_uniform'))
        model.add(Dense(8, activation='relu', init='lecun_uniform'))
        model.add(Dense(1, activation='linear', init='lecun_uniform'))

        model.compile(optimizer='adam', loss='mse')
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
        return 1+np.argmax(self.get_q_values(state))

    def get_q_values(self, state):
        """
        Returns Q values for all actions given a state.

        Arguments:
         state: state of the environment
        """
        q_values = np.zeros((self.action_size, ))
        for a in range(q_values.size):
            q_values[a] = self.model.predict(self.merge_state_action(1+a, state))
        return q_values

    def train(self, env):
        """
        Train agent in environment

        Arguments:
         env: gym environment
        """
        s_t = get_state(get_positions(env.reset()), add_direction=True)

        for iepisode in range(self.number_of_episodes):
            if self.verbose:
                print 'Episode', iepisode
            last_ball_position = np.array([np.NaN, np.NaN])

            replay_buffer = np.empty((self.dataset_size, self.state_size*(self.temporal_chain+1)+2))
            n_win = 0
            n_loss = 0

            states = [s_t]

            for j in range(self.dataset_size+self.temporal_chain-1):
                # select action a
                if random.random() > self.epsilon and len(states) == self.temporal_chain:
                    a_t = self.act(states)
                else:
                    distance = s_t[0] - s_t[3]
                    if distance > 0.5:
                        a_t = 2
                    elif distance < -0.5:
                        a_t = 3
                    else:
                        a_t = np.random.randint(1, 4)

                if self.verbose:
                    env.render()

                # take action and get reward
                (o_t1, r_t, done, _) = env.step(a_t)

                if r_t == 1:
                    n_win += 1
                elif r_t == -1:
                    n_loss += 1

                # get state
                positions = get_positions(o_t1, last_ball_position=last_ball_position)
                last_ball_position = positions['ball']
                if  positions['distance'] < 8:
                    r_t = 0.5
                s_t1 = get_state(positions, add_direction=True)
                if r_t == 0 and (s_t1[0] < -1 or s_t1[0] > 1):
                    r_t = -0.5

                # replay memory
                states.append(s_t1)
                if len(states) == self.temporal_chain+1:
                    replay_buffer[j-self.temporal_chain+1, :] = self.package_replay(a_t, r_t, states)
                    del states[0]

                s_t = s_t1

                # reset env
                if done:
                    s_t = get_state(get_positions(env.reset()), add_direction=True)

            self.buffer.add_equal(replay_buffer)
            minibatch = self.buffer.sample_equal(self.dataset_size)

            # create targets (no terminal state in pong)
            tts = np.zeros((minibatch.shape[0], 1))
            sss = np.zeros((minibatch.shape[0], s_t.size*self.temporal_chain+3))
            for i in range(0, tts.shape[0]):
                (aa, rr, states) = self.unpackage_replay(minibatch[i, :])
                sss[i, :] = self.merge_state_action(aa, states[:-1])

                qs = self.get_q_values(states[1:])
                tts[i] = rr + self.decay * np.max(qs)

            # train network
            self.model.fit(sss, tts, nb_epoch=10, batch_size=self.mb_size, verbose=False)
            self.model.save_weights(self.save_name + '.h5')

            # decay epsilon
            self.epsilon -= 1.0 / min(self.number_of_episodes, 1000)
            self.epsilon = max(self.min_epsilon, self.epsilon)
            if n_win == 0 and self.epsilon < 0.4:
                self.epsilon += 1.0 / min(self.number_of_episodes, 100)

            print 'Episode', iepisode, 'Epsilon', self.epsilon, 'Wins', n_win, 'Losses', n_loss


    def merge_state_action(self, a_t, states):
        assert a_t in [1, 2, 3]
        # a_t = 1 -> 0 0 1
        # a_t = 2 -> 0 1 0
        # a_t = 3 -> 1 0 0
        result = np.zeros((3,))
        result[-np.int(a_t)] = 1
        for i in range(len(states)):
            result = np.concatenate((result, states[i]))
        
        return result[None, :]

    def package_replay(self, a_t, r_t, states):
        state_size = states[0].size
        package = np.zeros((state_size*len(states) + 2))
        package[0] = a_t
        package[1] = r_t
        for i in range(len(states)):
            package[2+i*state_size:2+(i+1)*state_size] = states[i]
        return package[None, :]

    def unpackage_replay(self, package):
        state_size = (package.size-2.0)/(self.temporal_chain+1)
        assert state_size == np.int(state_size)
        state_size = np.int(state_size)
        a_t = package[0]
        r_t = package[1]
        states = []
        for i in range(self.temporal_chain+1):
            states.append(package[2+i*state_size:2+(i+1)*state_size])
        return (a_t, r_t, states)
