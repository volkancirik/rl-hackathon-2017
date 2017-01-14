import numpy as np

class experience_buffer():
    '''
    Basically a queue, with options to equally smaple rewards:
     - add:     np.arrays of size nxm
     - sample:  randomly smaple the buffer
     - sample_equal samples all reward types equally
    '''
    def __init__(self, buffer_size=2000, reward_index=5):
        self.buffer = np.empty((0,))
        self.buffer_size = buffer_size
        self.reward_index = reward_index

    '''
    Adds new expereicnes (np.array of size entries x length). It replaces the oldest entries if necessary
    '''
    def add(self, experience):
        number_to_remove = self.buffer.shape[0] + experience.shape[0] - self.buffer_size

        if number_to_remove > 0:
            self.buffer = self.buffer[number_to_remove:, :]

        if self.buffer.size == 0:
            self.buffer = np.empty(np.concatenate(([0], experience.shape[1:])))

        self.buffer = np.append(self.buffer, experience, axis=0)

    '''
    Samples the buffer randomly
    '''
    def sample(self, number_of_samples):
        return self.buffer[np.random.randint(0, self.buffer.shape[0], number_of_samples), :]

    '''
    Samples the buffer so that each reward has the same probability
    '''
    def sample_equal(self, number_of_samples):
        values = self.buffer[:, self.reward_index]
        types = np.unique(values)

        results = np.empty((number_of_samples, self.buffer.shape[1]))
        number_inserted = 0

        # sample all rewards equally
        for i in range(types.size):
            index = np.where(types[i] == values)[0]
            new_data = self.buffer[index[np.random.randint(0, index.size, np.int(np.floor(number_of_samples/types.size)))], :]
            results[number_inserted:number_inserted+new_data.shape[0], :] = new_data
            number_inserted += new_data.shape[0]

        # fill remaining slots
        if number_inserted < number_of_samples:
            results[number_inserted:, :] = self.sample(number_of_samples - number_inserted)

        return results
