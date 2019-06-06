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

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls


py.sign_in('ceniii', 'agt75cGwsrgQz14Jlr04')

def visualize(net, label, fig, cb, iterno):
	cbarlocs = [[0.9, 0.65],[0.35,0.1]]
	hm = net.getAttention(label).data.numpy()
	fig['data']=[]
	# plot here
	for i in range(2):
		# img = ax[0][i].imshow(hm[1+i*3])
		trace = go.Heatmap(z=hm[1+i*3],name='heatmap',colorbar=dict(len=0.25, title='hm'+str(i+1),y=cbarlocs[0][i],tickmode='array',tickvals=[0,2,4,6,8],ticktext=['0','2','4','6','8'])) #y=cbarlocs[0][i]
		fig.append_trace(trace,1,i+1)
		# if cb[0][i] is not None:
		# 	cb[0][i].remove()
		# cb[0][i] = plt.colorbar(img,ax=ax[0][i])

	hm = net.getGradAttention().data.numpy()

	# plot here
	for i in range(2):
		# img = ax[1][i].imshow(hm[1+i*3])
		# if cb[1][i] is not None:
		# 	cb[1][i].remove()
		# cb[1][i] = plt.colorbar(img,ax=ax[1][i])
		trace = go.Heatmap(z=hm[1+i*3],name='gradmap',colorbar=dict(len=0.25, title='gm'+str(i+1),y=cbarlocs[1][i])) #,y=cbarlocs[1][i]
		fig.append_trace(trace,2,i+1)

	# fig.suptitle('iteration '+str(iterno))
	fig['layout'].update(height=600, width=800, title=('iteration '+str(iterno)))
	plot_url = py.plot(fig, filename='simple-subplot-with-annotations',auto_open=False)
	print(plot_url)
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
		net = net.cuda()
	# fig, ax = plt.subplots(nrows=2, ncols=2)
	fig = tls.make_subplots(rows=2, cols=2)
	
	iterno = 0
	cb = [[None,None],[None,None]]
	while True:
		pred = net(data)
		loss = crit(pred, label)

		optimizer.zero_grad()
		loss.backward()

		if iterno%2==0:
			# todo: visualize attention map
			visualize(net,label,fig,cb,iterno)
		optimizer.step()
		# time.sleep(0.1)
		iterno += 1



if __name__ == '__main__':

	net = WeaklySupNet()
	optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
	crit = torch.nn.NLLLoss()
	data, label = loadData()
	train(net, data, label, optimizer, crit)
