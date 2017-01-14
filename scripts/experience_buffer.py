# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.vffy0mfk1
import numpy as np
import random

class experience_buffer():
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self, number_of_samples):
    	data = np.zeros((number_of_samples, self.buffer[0].size))
    	for i in range(0, number_of_samples):
    		index = np.random.randint(0, len(self.buffer))
    		data[i, :] = self.buffer[index]
        return data