import numpy as np
import torch
from resnet import resnet50
import os
from utils import load_state_dict_from_url

size = 224

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def load_model() :

	model = resnet50(pretrained = False)
	state_dict = load_state_dict_from_url(model_urls['resnet50'],
										  progress=True, model_dir="./")
	new_state_dict = {}
	for key in list(state_dict.keys()) :
		new_state_dict[key] = state_dict[key]
	#model.load_state_dict(new_state_dict)
	return model
	
if __name__ == "__main__" :
	model = load_model()

	x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

	out = model(x)

	print(out.size())


