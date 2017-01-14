import numpy as np
import pickle
import gym
import pdb

import keras.models as M
import keras.layers as L
import keras.preprocessing as P

import sys

from scripts.preprocess import get_positions, get_state
from scripts.experience_buffer import experience_buffer

#hyperparameters
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 ## decay rate factor for RMSProp
resume = False # resume from previous checkpoint
render = False

## model initialization
model = M.Sequential()
model.add(L.Dense(9, input_shape=(4,), activation='sigmoid'))
model.add(L.Dense(9, activation='sigmoid'))
#model.add(L.Dense(9, activation='relu'))
model.add(L.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

env = gym.make('Pong-v0')
s_t = get_state(get_positions(env.reset()))
xs,dlogps,drs,act = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

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
    
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

wins =0
total = 0
action_word = {1:'stay', 2:'up', 3:'down'}
while(True):
    if(render): env.render()

    ## forward pass
    aprob = model.predict(np.matrix(s_t))
    action = action_sample(aprob)
    
    xs.append(s_t) #observed state
    dlogps.append(aprob) ## probability
    act.append(action-1)
    
    ## step in the environment
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)

    if (reward !=0):
        total+=1
        print(action_word[action])
    if(reward == 1):
        wins+=1
        print('Episode No: %5d  Reward: %d, Win/Loss: %d/%d' % (episode_number, reward, wins, total))
    
    if done: # an episode finished
        ## train for k epochs
        episode_number+=1
        #pdb.set_trace()
        env.reset()

        if(episode_number % batch_size == 0):
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
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            
            model.fit(x=epx,y=discounted_epr*eact, verbose=False)

            
    

