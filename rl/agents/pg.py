import random
import os

import numpy as np
from matplotlib import pyplot as plt

import pickle
import gym
import pdb

import keras.models as M
import keras.layers as L
import keras.preprocessing as P

import sys

from agent import Agent
from ..scripts.preprocess import get_positions, get_state
from ..scripts.experience_buffer import experience_buffer


class PGagent(Agent):
    """
    Policy Gradient Learning Agent based on http://karpathy.github.io/2016/05/31/rl/

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
      load: Boolean value to decide if you want to load previously saved models or not
      render: Boolean value to deicide if you want to render the gameplay
      gamma: Discounted Reward factor
    """
    def __init__(self, mb_size=32, save_name='pg', dataset_size=2000,\
                 state_size=4, action_size=6,  \
                 epsilon = 1.0, min_epsilon=0.1, decay=0.9, number_of_episodes=100000, \
                 verbose = False, load=True, render=False, batch_size=10, gamma = 0.99):
        self.mb_size = mb_size
        self.save_name = save_name
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.verbose = verbose
        self.render = render
        self.batch_size = batch_size
        self.gamma = gamma

        self.decay = decay
        self.min_epsilon = min_epsilon
        self.number_of_episodes = number_of_episodes
        self.dataset_size = dataset_size

        self.build_model(load)

    # Function to calculate discounted rewards if a vector of rewards for the complete episode are given
    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    # Build the policy gradient model
    def build_model(self, load):
        """
        Builds neural networks. Loads previous weights autoamtically.
        """
        ## model initialization
        model = M.Sequential()
        model.add(L.Dense(9, input_shape=(4,), activation='sigmoid'))
        model.add(L.Dense(9, activation='sigmoid'))
        #model.add(L.Dense(9, activation='relu'))
        model.add(L.Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        if (load):
            try:
                model.load_weights(self.save_name + '.h5')
                print ('Loaded saved model')
            except:
                print('Coudn\'t find the model. Initialising randomly')

        self.model = model

    # given the previous state and action, predict the next action
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
        aprob = self.model.predict(np.matrix(state))
        action = action_sample(aprob)

        return action, aprob

    def train(self, env):
        """
        Train agent in environment

        Arguments:
         env: gym environment
        """

        # Initialize hyperparameters
        s_t = get_state(get_positions(env.reset()))
        xs,dlogps,drs,act = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0

        wins =0
        total = 0
        action_word = {1:'stay', 2:'up', 3:'down'}

        # Training loop
        while(True):
            if(self.render): env.render()

            ## forward pass
            action, aprob = self.act(s_t)
            xs.append(s_t) #observed state
            dlogps.append(aprob) ## probability
            act.append(action-1)

            ## step in the environment
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            drs.append(reward)

            # printing results
            if (reward !=0):
                total+=1
                if self.verbose:
                    print(action_word[action])
            if(reward == 1):
                wins+=1
                print('Episode No: %5d  Reward: %d, Win/Loss: %d/%d' % (episode_number, reward, wins, total))

            if done: # an episode finished
                ## train for k epochs
                episode_number+=1
                print('Episode No: %d' %(episode_number))
                env.reset()

                if(episode_number % self.batch_size == 0):
                    epx = np.vstack(xs)
                    epdlogp = np.vstack(dlogps)
                    epr = np.vstack(drs)

                    eact = np.vstack(act)
                    n_values = 3
                    eact_temp = np.zeros((eact.shape[0],n_values))
                    eact_temp[np.arange(eact.shape[0]),eact.T]=1
                    eact = eact_temp

                    xs, dlogps, drs, act = [],[],[],[] # reset arrays
                    
                    # compute the discounted rewards backwards
                    discounted_epr = self.discount_rewards(epr)
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)

                    # train the policy gradient model
                    self.model.fit(x=epx,y=discounted_epr*eact, verbose=False)
                    self.model.save(self.save_name + '.h5')

# Given the probablity distribution of the actions, sample one of them
def action_sample(aprob):
    cumm_aprob = np.zeros_like(aprob)
    cumm_aprob[0,0] = aprob[0,0]
    sample = np.random.uniform()
    if(sample < cumm_aprob[0,0]):
        return 1
    for i in range(1,aprob.shape[1]):
        cumm_aprob[0,i] = cumm_aprob[0,i-1] + aprob[0,i]
        if(sample < cumm_aprob[0,i]):
            return i+1

