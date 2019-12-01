import numpy as np
import torch
from AlexNet import create_model
import os

def load_model() :
	opt = {'num_classes':4}
	
	net = create_model(opt)
	
	class Network(object) :
		def __init__(self, net, exp_dir) :
			self.networks = {"model":net}
			self.exp_dir = exp_dir
	
		def load_checkpoint(self, epoch, train=True, suffix=''):
	
			for key, net in self.networks.items(): # Load networks
				self.load_network(key, epoch,suffix)
	
			self.curr_epoch = epoch
	
		def load_network(self, net_key, epoch,suffix=''):
			assert(net_key in self.networks)
			filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
			assert(os.path.isfile(filename))
			if os.path.isfile(filename):
				checkpoint = torch.load(filename)
				self.networks[net_key].load_state_dict(checkpoint['network'])
	
		def _get_net_checkpoint_filename(self, net_key, epoch):
			return os.path.join(self.exp_dir, net_key+'_net_epoch'+str(epoch))
	
	print("returning the model")
	net_wrap = Network(net, "./")
	net_wrap.load_checkpoint(50, train = False)
	net = net_wrap.networks["model"]
	return net

if __name__ == "__main__" :
	net = load_model()
	size = 224
	x = torch.autograd.Variable(torch.FloatTensor(2,3,size,size).uniform_(-1,1))
	
	out = net(x, out_feat_keys=["fc_block"])
	#out = net(x, out_feat_keys=net.all_feat_names)
	print(out.size())
	print(out[1].size())
	for f in range(len(out)):
		print('Output feature {0} - size {1}'.format(
			net.all_feat_names[f], out[f].size()))
	
	filters = net.get_L1filters()
	
	print('First layer filter shape: {0}'.format(filters.size()))
