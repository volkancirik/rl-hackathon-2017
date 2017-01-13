import sys
sys.path.append('../scripts')

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_erosion

import gym

from keras.models import Sequential
from keras.layers import Dense

from preprocess import get_positions, get_state

def run():
    # init keras
    model = init_model()

    # init environment
    # 0,1 -> stay
    # 2,4 -> up
    # 3,5 -> down
    env = gym.make('Pong-v0')
    s_t = get_state(get_positions(env.reset()))
    for _ in range(1000):

        # select action a
        # argmax a Q(state, action)
        a_t = np.argmax(model.predict(s_t[None, :]))
        if a_t != 0: # up/down
            a_t = a_t + 1

        # take action and get reward
        env.render()
        (o_t1, r_t, _, _) = env.step(a_t)

        # get state
        pos_t1 = get_positions(o_t1)
        s_t1 = get_state(pos_t1)

        # replay memory

        # create targets

        # train network

        s_t = s_t1


def init_model():
    model = Sequential()
    model.add(Dense(9, input_shape=(3,), activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(3, activation='linear'))

    model.compile(optimizer='rmsprop', loss='mse')

    return model

if __name__ == '__main__':
    run()
