import numpy as np

class experience_buffer():
    '''
    Basically a queue, with options to equally smaple rewards:
     - add:     np.arrays of size nxm
     - sample:  randomly smaple the buffer
     - sample_equal samples all reward types equally
    '''
    def __init__(self, buffer_size=2000, reward_index=5, action_index=4):
        self.buffer = np.empty((0,))
        self.buffer_size = buffer_size
        self.reward_index = reward_index
        self.action_index = action_index

    def almost_full(self):
        return self.buffer.shape[0] >= self.buffer_size*0.75

    '''
    Adds new experiences (np.array of size entries x length).
    It replaces the oldest entries if necessary
    '''
    def add(self, experience):
        number_to_remove = self.buffer.shape[0] + experience.shape[0] - self.buffer_size

        if number_to_remove > 0:
            self.buffer = self.buffer[number_to_remove:, :]

        if self.buffer.size == 0:
            self.buffer = np.empty(np.concatenate(([0], experience.shape[1:])))

        self.buffer = np.append(self.buffer, experience, axis=0)


    '''
    Adds new experiences (np.array of size entries x length).
    It replaces the the most frequent rewards&action first.
    '''
    def add_equal(self, experience):

        experience = unique_entries(experience)

        number_to_remove = self.buffer.shape[0] + experience.shape[0] - self.buffer_size

        while number_to_remove > 0:
            # remove most frequent reward
            values = self.buffer[:, [self.reward_index, self.action_index]]
            types = unique_entries(values)
            hist = hist_fun(values, types)
            print hist
            most_frequent_index = np.all(types[np.argmax(hist)] == values, 1)
            number_of_most_frequent_reward = np.sum(most_frequent_index)
            remove_index = np.where(most_frequent_index)[0][0:min(number_of_most_frequent_reward, number_to_remove)]
            keep_index = np.setdiff1d(np.arange(self.buffer.shape[0]), remove_index)
            self.buffer = self.buffer[keep_index, :]
            number_to_remove -= remove_index.size

        if self.buffer.size == 0:
            self.buffer = np.empty(np.concatenate(([0], experience.shape[1:])))

        self.buffer = np.append(self.buffer, experience, axis=0)
        self.buffer = unique_entries(self.buffer)

    '''
    Samples the buffer randomly
    '''
    def sample(self, number_of_samples):
        return self.buffer[np.random.randint(0, self.buffer.shape[0], number_of_samples), :]

    '''
    Samples the buffer so that each reward&action has the same probability
    '''
    def sample_equal(self, number_of_samples):
        values = self.buffer[:, [self.reward_index, self.action_index]]
        types = unique_entries(values)

        results = np.empty((number_of_samples, self.buffer.shape[1]))
        number_inserted = 0

        # sample all rewards equally
        for i in range(types.shape[0]):
            index = np.where(np.all(types[i, :] == values, 1))[0]
            new_data = self.buffer[index[np.random.randint(0, index.size, np.int(np.floor(number_of_samples/types.size)))], :]
            results[number_inserted:number_inserted+new_data.shape[0], :] = new_data
            number_inserted += new_data.shape[0]

        # fill remaining slots
        if number_inserted < number_of_samples:
            results[number_inserted:, :] = self.sample(number_of_samples - number_inserted)

        return results

def unique_entries(data):
    b = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
    _, idx = np.unique(b, return_index=True)

    return data[idx]

def hist_fun(data, x):
    result = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = np.sum(np.all(data == x[i, :], 1))
    return result