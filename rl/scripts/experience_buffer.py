# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.vffy0mfk1
import numpy as np

class experience_buffer():
    def __init__(self, buffer_size=1000, reward_index=5):
        self.buffer = np.empty((0,))
        self.buffer_size = buffer_size
        self.reward_index = reward_index

    def add(self, experience):
        number_to_remove = self.buffer.shape[0] + experience.shape[0] - self.buffer_size

        while number_to_remove > 0:
            # remove most frequent reward
            values = self.buffer[:, self.reward_index]
            types = np.unique(values)
            (hist, _) = np.histogram(values, bins=types.size)
            print hist
            most_frequent_index = types[np.argmax(hist)] == values
            number_of_most_frequent_reward = np.sum(most_frequent_index)
            remove_index = np.where(most_frequent_index)[0][0:min(number_of_most_frequent_reward, number_to_remove)]
            keep_index = np.setdiff1d(np.arange(self.buffer.shape[0]), remove_index)
            self.buffer = self.buffer[keep_index, :]
            number_to_remove -= remove_index.size

        if self.buffer.size == 0:
            self.buffer = np.empty(np.concatenate(([0], experience.shape[1:])))
        self.buffer = np.append(self.buffer, experience, axis=0)

    def sample(self, number_of_samples):
        return self.buffer[np.random.randint(0, len(self.buffer), number_of_samples), :]
