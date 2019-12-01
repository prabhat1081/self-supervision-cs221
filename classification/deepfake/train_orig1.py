import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

from datamgr import SimpleDataManager
from torch.autograd import Variable
from load_model_orig import load_model
import sys

base_file = "train.json"
val_file = "test.json"

image_size = 224

checkpoint_dir = "models"
if not os.path.exists(checkpoint_dir) :
	os.makedirs(checkpoint_dir)


base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
base_loader     = base_datamgr.get_data_loader( base_file , aug = True )
val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
val_loader      = val_datamgr.get_data_loader( val_file, aug = False)

loss_fn = nn.CrossEntropyLoss()

def train_loop(base_loader, model, classifier, rotate_classifier, optimizer, epoch) :
	avg_loss = 0
	print_freq = 10
	model.train()
	

	for i, (x, y) in enumerate(base_loader) :
		x = Variable(x).cuda()
		y = Variable(y).cuda()
		bs = x.size(0)
		x_ = []
		y_ = []
		a_ = []
		for j in range(bs) :
			x90 = x[j].transpose(2,1).flip(1)
			x180 = x90.transpose(2,1).flip(1)
			x270 =  x180.transpose(2,1).flip(1)
			x_ += [x[j], x90, x180, x270]
			y_ += [y[j] for _ in range(4)]
			a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]
		
		x_ = Variable(torch.stack(x_,0)).cuda()
		y_ = Variable(torch.stack(y_,0)).cuda()
		a_ = Variable(torch.stack(a_,0)).cuda()

		feats = model(x_)
		out = classifier(feats)
		rotate_scores = rotate_classifier(feats)

		optimizer.zero_grad()
		loss = (loss_fn(out, y_) + loss_fn(rotate_scores, a_))/2.0
		loss.backward()
		optimizer.step()

		avg_loss = avg_loss+loss.item()
		if i % print_freq==0:
			print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(base_loader), avg_loss/float(i+1)  ))


def validate_loop(val_loader, model, classifier, epoch) :
	model.eval()
	correct = 0
	total = 0
	for i, (x, y) in enumerate(val_loader) :
		x = Variable(x).cuda()
		y = Variable(y).cuda()
		feats = model(x)
		out = classifier(feats)

		pred = out.cpu().data.numpy().argmax(axis = 1)
		correct += (pred == y.cpu().data.numpy()).sum()
		total += len(pred)
	acc = 100.0 * float(correct)/float(total)
	print('Epoch {:d} | Acc {:f}'.format(epoch, acc))
	return acc

def save_checkpoint(epoch, model, checkpoint_dir) :
	outfile = os.path.join(checkpoint_dir, 'best_model1.tar')
	torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)


model = load_model()
model = model.cuda()

classifier = nn.Sequential(nn.Linear(4096,2))
classifier = classifier.cuda()

rotate_classifier = nn.Sequential(nn.Linear(4096,4))
rotate_classifier = rotate_classifier.cuda()


optimizer = torch.optim.Adam([{'params':classifier.parameters()},{'params':rotate_classifier.parameters()}], lr = 2e-4)

start_epoch = 0
end_epoch = 5

best_acc = -1.0

for epoch in range(start_epoch, end_epoch) :

	train_loop(base_loader, model, classifier, rotate_classifier, optimizer, epoch)
	acc = validate_loop(val_loader, model, classifier, epoch)
	if acc > best_acc :
		best_acc = acc
		save_checkpoint(epoch, model, checkpoint_dir)




