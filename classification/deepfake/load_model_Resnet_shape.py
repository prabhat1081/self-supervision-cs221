import numpy as np
import torch
from resnet import resnet50
import os
from utils import load_state_dict_from_url
from torch.utils import model_zoo

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
    'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    'vgg16_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar',
    'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
}

def load_model() :

	model = resnet50()
	# model = torch.nn.DataParallel(model).cuda()

	# state_dict = load_state_dict_from_url(model_urls['resnet50_trained_on_SIN'],
										  # progress=True, model_dir="./")
	# new_state_dict = {}
	# for key in list(model.state_dict().keys()) :
	# 	new_state_dict[key] = state_dict[key]
	# model.load_state_dict(state_dict)
	model = torch.nn.DataParallel(model).cuda()
	checkpoint = model_zoo.load_url(model_urls['resnet50_trained_on_SIN'])
	model.load_state_dict(checkpoint["state_dict"])

	return model
	
if __name__ == "__main__" :
	model = load_model()

	x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

	out = model(x)

	print(out.size())


