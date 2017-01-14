"""
abstract class for agent
TODO : explain functions, arguments
"""
class Agent(object):
	def __init__(self, mb_size = 32, save_name = 'abs' , state_size = None, action_size = None,  epsilon = 0.1, verbose = True):
		self.mb_size = mb_size
		self.save_name = save_name
		self.state_size = state_size
		self.action_size = action_size
		self.epsilon = epsilon
		self.verbose = verbose
		self.build_model()

	def build_model(self):
		return
	def act(self, state):
		return
	def train(self, env):
		return
