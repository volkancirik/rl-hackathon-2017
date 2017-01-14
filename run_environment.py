"""
run a sample pong environment.

TODO :
 - comment each line
 - log
"""
import gym
from rl.agents.ac import *
from rl.agents.dqn import *
from rl.agents.pg import *

MY_ENV_NAME = 'Pong-v0'
env = gym.make(MY_ENV_NAME)
OBSERVATION_SPACE = 4 ## could be more than that with concatenation of previous states
ACTION_SPACE = env.action_space.n
METHODS = {'dqn': QLearningAgent, 'ac': ActorCriticAgent, 'pg': PGagent}
agent_type = 'pg'

agent = METHODS[agent_type](state_size = OBSERVATION_SPACE, action_size = ACTION_SPACE)
agent.train(env)
