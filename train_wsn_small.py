import torch
from torch.autograd import Variable
import tqdm
import numpy as np

if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device

import time
from attention.wsnet import WeaklySupNet#, Criterion

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def visualize(net, label, fig, ax, cb, iterno):
	hm = net.getAttention(label).data.numpy()

	# plot here
	for i in range(len(ax[0])):
		img = ax[0][i].imshow(hm[2+i*3])
		if cb[0][i] is not None:
			cb[0][i].remove()
		cb[0][i] = plt.colorbar(img,ax=ax[0][i])

	hm = net.getGradAttention().data.numpy()

	# plot here
	for i in range(len(ax[1])):
		img = ax[1][i].imshow(hm[2+i*3])
		if cb[1][i] is not None:
			cb[1][i].remove()
		cb[1][i] = plt.colorbar(img,ax=ax[1][i])

	fig.suptitle('iteration '+str(iterno))
	plt.pause(0.05)
	return cb

def loadData():
	filelist = os.listdir('./data')
	imgs = []
	for file in filelist:
		imgs.append(np.moveaxis(cv2.imread(os.path.join('./data/',file)),-1,0))
	label = [0,0,0,1,1,1]

	imgs = torch.tensor(imgs).type(device.FloatTensor)
	label = torch.tensor(label)

	return imgs, label


def train(net, data, label, optimizer, crit, epoches=100):
	if torch.cuda.is_available():
		data = data.cuda()
		net = net.cuda()
		crit = crit.cuda()
	fig, ax = plt.subplots(nrows=2, ncols=2)
	
	iterno = 0
	cb = [[None,None],[None,None]]
	while True:
		pred = net(data)
		loss = crit(pred, label)

		optimizer.zero_grad()
		loss.backward()

		if iterno%2==0:
			# todo: visualize attention map
			visualize(net,label,fig,ax,cb,iterno)
		optimizer.step()
		# time.sleep(0.1)
		iterno += 1



if __name__ == '__main__':

	net = WeaklySupNet()
	optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
	crit = torch.nn.NLLLoss()
	data, label = loadData()
	train(net, data, label, optimizer, crit)
