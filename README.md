# rl-hackathon-2017

## Pre-requisites

### Resources
This repo assumes familiarity in ML in general(algorithms, tools). Our starting point was reading these well-written introductory posts on Reinforcement Learning. We highly recommend reading these posts before moving forward.

- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- [Q-Learning And Exploration](https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/)
- [Learning Reinforcement Learning](http://www.wildml.com/2016/10/learning-reinforcement-learning/)
- [Action-Value Methods and n-armed bandit problems](http://outlace.com/Reinforcement-Learning-Part-1/)
- [Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
- [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)

In addition to these awesome resources, there are bunch of open-source RL implementations out there, we make use of following resources as starting points and built on top of them:
- [Actor Critic with OpenAI Gym¶](https://github.com/gregretkowski/notebooks/blob/master/ActorCritic-with-OpenAI-Gym.ipynb)
- [DQN with Experience Replay](https://github.com/sherjilozair/dqn)
- [Stochastic Policy Gradients on Pong](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)

### Tools
To move quickly we make use of following tools:
- [OpenAI's Gym](https://gym.openai.com/docs) for experimantal environment. Before writing a single line of code, we recommend reading the documentation.
- [Keras](https://keras.io/) for quicky prototyping neural network models.
- [Jupyter Notebooks](http://jupyter.org/) for fast and interactive development. 

## Algorithms

### Q-Learning

### Vanilla Policy Gradients

### Actor-Critic Policy Gradients


## Creating A New Enrivornment:
If you would like to exploit the Gym for a target task in your mind, you can create a new environment. For an example, we would like to create an environment for a simplified computer vision problem.
We would like to localize a target object in a scene. Here agent can move a frame and the aim is to find the object in this scene. 

![Object Localization](http://research.microsoft.com/en-us/um/people/jingdw/salientobjectdetection/Salient%20Object%20Detection_files/2_reg.jpg "Object Localization Example")

Starting from [CartPole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) implementation, we implemented a simple environment. All you need to figure out how to draw simple objects. 
You can find [new environment code](rl/environments/object_localization.py) and an [example notebook](notebooks/new_environment.ipynb) under this repository. You place `object_localization.py` under `/gym/envs/classic_control/`. Also, you need to edit a couple of initialization files as explained [here](https://github.com/openai/gym/wiki/Environments). Here's an agent in action in this environment:


![Random Agent](screenshots/gif-random.gif)

![Agent Moving to Object](screenshots/gif-moving.gif)
