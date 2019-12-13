import numpy as np
import torch
from AlexNet_orig import AlexNet
import os
from utils import load_state_dict_from_url

size = 224

model_urls = {
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def load_model() :

	model = AlexNet()

	state_dict = load_state_dict_from_url(model_urls['alexnet'],
										  progress=True, model_dir="./")
	new_state_dict = {}
	for key in list(model.state_dict().keys()) :
		new_state_dict[key] = state_dict[key]
	model.load_state_dict(new_state_dict)
	return model
	
if __name__ == "__main__" :
	model = load_model()

	x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

	out = model(x)

	print(out.size())


