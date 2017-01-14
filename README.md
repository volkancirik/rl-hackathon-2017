# rl-hackathon-2017



### Creating A New Enrivornment:
If you would like to exploit the Gym for a target task in your mind, you can create a new environment. For an example, we would like to create an environment for a simplified computer vision problem.
We would like to localize a target object in a scene. Here agent can move a frame and the aim is to find the object in this scene. 

![Object Localization](http://research.microsoft.com/en-us/um/people/jingdw/salientobjectdetection/Salient%20Object%20Detection_files/2_reg.jpg "Object Localization Example")

Starting from [CartPole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) implementation, we implemented a simple environment. All you need to figure out how to draw simple objects. 
You can find new environment code and an example notebook under this repository. You place `object_localization.py` under `/gym/envs/classic_control/`. Also, you need to edit a couple of initialization files as explained [here](https://github.com/openai/gym/wiki/Environments). Here's an agent in action in this environment:


![Random Agent](screenshots/gif-random.gif)

![Agent Moving to Object](screenshots/gif-moving.gif)
