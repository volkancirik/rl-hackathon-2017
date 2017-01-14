import sys
sys.path.append('../scripts')

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_erosion

import gym

from keras.models import Sequential
from keras.layers import Dense

from preprocess import get_positions, get_state
from experience_buffer import experience_buffer

number_of_actions = 6

def run():
    # init keras
    model = init_model()
    buffer = experience_buffer(buffer_size = 10000)
    batch_size = 100
    decay = 0.5;

    # init environment
    # 0,1 -> stay
    # 2,4 -> up
    # 3,5 -> down
    env = gym.make('Pong-v0')
    s_t = get_state(get_positions(env.reset()))

    for _ in range(1000):

        for _ in range(batch_size*1):
            # select action a
            # argmax a Q(state, action)
            a_t = np.argmax(get_q_values(model, s_t))

            # take action and get reward
            env.render()
            (o_t1, r_t, done, _) = env.step(a_t)

            # get state
            pos_t1 = get_positions(o_t1)
            s_t1 = get_state(pos_t1)

            # replay memory
            buffer.add(pacakge_replay(s_t, a_t, r_t, s_t1))

            s_t = s_t1

            # reset env
            if done:
                s_t = get_state(get_positions(env.reset()))
        
        minibatch = buffer.sample(batch_size)

        # create targets (no terminal state in pong)
        tts = np.zeros((minibatch.shape[0], 1))
        sss = np.zeros((minibatch.shape[0], s_t.size+1))
        for i in range(0, tts.shape[0]):
            (ss, aa, rr, ss_) = unpacakge_replay(minibatch[i, :])
            qs = get_q_values(model, ss_)
            tts[i] = rr + decay * np.max(qs)
            sss[i, :] = merge_state_action(ss, aa)

        # train network
        model.fit(sss, tts, nb_epoch=100, batch_size=20)

    env.close()

def merge_state_action(s_t, a_t):
    input = np.zeros((1, s_t.size+1))
    input[-1] = a_t
    return input

def get_q_values(model, s_t):
    q_values = np.zeros((number_of_actions, ))
    for i in range(q_values.size):
        input = merge_state_action(s_t, i)
        q_values[i] = model.predict(input)

    return q_values

def pacakge_replay(s_t, a_t, r_t, s_t1):
    package = np.zeros((s_t.size*2 + 2))
    package[0:s_t.size] = s_t
    package[s_t.size+1] = a_t
    package[s_t.size+2] = r_t
    package[-s_t.size:] = s_t1
    return package[None, :]

def unpacakge_replay(package):
    size = (package.size-2)/2
    s_t = package[:size]
    a_t = package[size+1]
    r_t = package[size+2]
    s_t1 = package[-size:]
    return (s_t, a_t, r_t, s_t1)

def init_model():
    model = Sequential()
    model.add(Dense(9, input_shape=(4+1,), activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='rmsprop', loss='mse')

    return model

if __name__ == '__main__':
    run()
