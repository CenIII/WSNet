"""
high level support for doing this and that.
"""
import torch
from torch.autograd import Variable
import tqdm
import numpy as np
from utils.loss import multilabel_soft_pull_loss

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
import matplotlib
matplotlib.use('tkagg')

def visualize(net, label, fig, ax, cb, iterno):
	hm = net.getAttention(label).data.cpu().numpy()

	# plot here
	for i in range(len(ax[0])):
		img = ax[0][i].imshow(hm[1+i*4])
		if cb[0][i] is not None:
			cb[0][i].remove()
		cb[0][i] = plt.colorbar(img,ax=ax[0][i])

	

	# plot here
	for i in range(len(ax[1])):
		hm = net.diffuse.drawDiffuseMap(1+i*4).data.cpu().numpy()
		img = ax[1][i].imshow(hm)
		if cb[1][i] is not None:
			cb[1][i].remove()
		cb[1][i] = plt.colorbar(img,ax=ax[1][i])
	

	hm = net.getTMask().data.cpu().numpy()
	for i in range(len(ax[2])):
		img = ax[2][i].imshow(hm[1+i*4])
		if cb[2][i] is not None:
			cb[2][i].remove()
		cb[2][i] = plt.colorbar(img,ax=ax[2][i])
	
	# get 8th image's both heatmaps
	# hm = net.heatmaps[8].data.cpu().numpy()
	# for i in range(len(ax[2])):
	# 	img = ax[2][i].imshow(hm[:,:,i+1])
	# 	if cb[2][i] is not None:
	# 		cb[2][i].remove()
	# 	cb[2][i] = plt.colorbar(img,ax=ax[2][i])

	fig.suptitle('iteration '+str(iterno))
	plt.pause(0.05)
	return cb

def loadData():
	filelist = sorted(os.listdir('./data'))
	imgs = []
	for file in filelist:
		imgs.append(np.moveaxis(cv2.imread(os.path.join('./data/',file)),-1,0))
	label = [[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]]
	label_vis = [0,0,0,0,1,1,1,1]
	imgs = torch.tensor(imgs).type(device.FloatTensor)
	label = torch.tensor(label).type(device.FloatTensor)
	label_vis = torch.tensor(label_vis)#.type(device.FloatTensor)
	return imgs, label, label_vis


def gauss_filt(data): #[8, 3, 512, 512]
	# mask
	zzz=data.sum(1,True)
	zzz[zzz>0]=1
	mask = 1-zzz.repeat(1,3,1,1)
	# generate gaussian noise
	N,C,H,W = data.shape
	mean = 128
	var = 1000
	sigma = var**0.5
	gauss = np.clip(np.around(np.random.normal(mean,sigma,(int(N/2),C,1,1)),decimals=0),0,255)
	gauss = torch.from_numpy(gauss).type(device.FloatTensor).repeat(2,1,H,W)
	# apply gaussian noise
	ret = data + gauss * mask
	ret = (ret-128.)/255.

	return ret


def train(net, data, label, label_vis, optimizer, crit0, crit1, epoches=100):
	if torch.cuda.is_available():
		data = data.cuda()
		net = net.cuda()
		crit0 = crit0.cuda()
		# crit1 = crit1.cuda()
		label = label.cuda()
		label_vis = label_vis.cuda()
	fig, ax = plt.subplots(nrows=3, ncols=2)
	
	iterno = 0
	cb = [[None,None],[None,None],[None,None]]
	filt_data = [gauss_filt(data) for i in range(10)]
	while True:
		if iterno%10==0:
			indlist = np.random.choice(10,10,replace=False)
		idx = np.arange(8)#np.random.choice(8,8,replace=False)
		pred,pred1,pred_tmask,_,_ = net(filt_data[indlist[iterno%10]][idx],label_vis[idx])
		loss = crit0(pred, label[idx])
		loss += crit0(pred1, label[idx])
		loss += crit1(pred_tmask, None)
		optimizer.zero_grad()
		loss.backward()
		print('iterno='+str(iterno)+', loss='+str(loss))

		if iterno%5==0:
			# todo: visualize attention map
			visualize(net,label_vis[idx],fig,ax,cb,iterno)
		optimizer.step()
		# time.sleep(0.1)
		iterno += 1



if __name__ == '__main__':

	net = WeaklySupNet(nclass=2)
	optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
	crit0 = torch.nn.MultiLabelSoftMarginLoss()
	crit1 = multilabel_soft_pull_loss
	data, label, label_vis = loadData()
	train(net, data, label, label_vis, optimizer, crit0, crit1)
