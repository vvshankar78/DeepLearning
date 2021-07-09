import pprint

class ModelConfig(object):

	def __init__(self,):
		super(ModelConfig, self).__init__()
		self.seed = 1
		self.batch_size_cuda = 512
		self.batch_size_cpu = 512
		self.num_workers = 4
		self.epochs = 24
		self.peak = 5
		self.dropout_value = 0.05

	def print_config(self):
		print("Model Parameters:")
		pprint.pprint(vars(self), indent=2)
