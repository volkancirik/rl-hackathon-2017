import numpy as np
import pickle
import gym
import pdb

import keras.models as M
import keras.layers as L

import sys
sys.path.append('../scripts')

from preprocess import get_positions, get_state
from experience_buffer import experience_buffer

#hyperparameters
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 ## decay rate factor for RMSProp
resume = False # resume from previous checkpoint
render = True

## model initialization
model = M.Sequential()
model.add(L.Dense(9, input_shape(4), activation='relu')
model.add(L.Dense(9, activation='relu'))
model.add(L.Dense(9, activation='relu'))
model.add(L.Dense(3, activation='softmax'))
model.compile(optimizer='adam')

env = gym.make('Pong-v0')
s_t = get_state(get_positions(env.reset()))

while(True):
    if(render):
        env.render()

    
    if done: # an episode finished
          
    

