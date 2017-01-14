import sys
sys.path.append('../scripts')

import numpy as np
from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense

from preprocess import get_positions, get_state
from experience_buffer import experience_buffer

number_of_actions = 6
length_of_states = 4

def run():
    # init keras
    model = init_model()
    main_buffer = experience_buffer(buffer_size=1000, reward_index=5)
    batch_size = 100
    decay = 0.9

    # init environment
    # 0,1 -> stay
    # 2,4 -> up
    # 3,5 -> down
    env = gym.make('Pong-v0')
    s_t = get_state(get_positions(env.reset()))

    for i in range(1000):
        print 'Episode', i

        for _ in range(5):
            replay_buffer = np.empty((batch_size, length_of_states*2+2))

            for j in range(batch_size):# better than random; stack states; read about q learning
                # select action a
                if i < 50:
                    if np.random.randint(0, 5) == 0:
                        a_t = np.argmax(get_q_values(model, s_t))
                        print 'argmax:', a_t
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
                            if np.random.randint(0, 2) == 0:
                                a_t = 0
                            else:
                                a_t = 1
                else:
                    # argmax a Q(state, action)
                    a_t = np.argmax(get_q_values(model, s_t))

                # take action and get reward
                env.render()
                (o_t1, r_t, done, _) = env.step(a_t)

                # get state
                s_t1 = get_state(get_positions(o_t1))

                # replay memory
                replay_buffer[j, :] = pacakge_replay(s_t, a_t, r_t, s_t1)

                s_t = s_t1

                # reset env
                if done:
                    s_t = get_state(get_positions(env.reset()))

            main_buffer.add(replay_buffer)

        minibatch = main_buffer.sample(batch_size)

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
    merged = np.zeros((1, s_t.size+1))
    merged[-1] = a_t
    return merged

def get_q_values(model, s_t):
    q_values = np.zeros((number_of_actions, ))
    for i in range(q_values.size):
        q_values[i] = model.predict(merge_state_action(s_t, i))

    return q_values

def pacakge_replay(s_t, a_t, r_t, s_t1):
    package = np.zeros((s_t.size*2 + 2))
    package[0:s_t.size] = s_t
    package[s_t.size] = a_t
    package[s_t.size+1] = r_t
    package[-s_t.size:] = s_t1
    return package[None, :]

def unpacakge_replay(package):
    size = (package.size-2)/2
    s_t = package[:size]
    a_t = package[size]
    r_t = package[size+1]
    s_t1 = package[-size:]
    return (s_t, a_t, r_t, s_t1)

def init_model():
    model = Sequential()
    model.add(Dense(6, input_shape=(length_of_states+1,), activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='rmsprop', loss='mse')

    return model

if __name__ == '__main__':
    run()
