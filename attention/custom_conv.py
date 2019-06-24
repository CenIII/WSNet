import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices
import math

import matplotlib.pyplot as plt

if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device

class CustmConv(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size=5,stride=1,dilation=1):
        super(CustmConv,self).__init__()  
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.ksize = kernel_size
        self.pad = int(kernel_size/2)*dilation
        self.dilation = dilation
        self.stride = stride
        self.Wt = torch.nn.Parameter(data=torch.randn(outchannel,inchannel,self.ksize,self.ksize)*0.01, requires_grad=True)
        # self.Wt.data.uniform_(-1, 1)
        self.bias = torch.nn.Parameter(data=torch.zeros(outchannel,1), requires_grad=True)
        # self.b.data.uniform_(-1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.Wt)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x): # NxDxHxW
        N,D,H,W = x.shape # 8,32,124,124
        Hf = self.ksize
        Wf = self.ksize
        # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
        X_col = im2col_indices(x, Hf, Wf, padding=self.pad, stride=self.stride, dilation=self.dilation)
        # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
        W_col = self.Wt.view(self.outchannel, -1)

        # 20x9 x 9x500 = 20x500
        out = W_col @ X_col + self.bias

        # Reshape back from 20x500 to 5x20x10x10
        # i.e. for each of our 5 images, we have 20 results with size of 10x10
        out = out.view(self.outchannel, H, W, N)
        out = out.permute(3, 0, 1, 2)

        return out


class CalibConv(CustmConv):
    def __init__(self,inchannel,outchannel,kernel_size=5,stride=1,dilation=1,nclass=2):
        super(CalibConv,self).__init__(inchannel,outchannel,kernel_size,stride,dilation)  
        self.linear_final = nn.Linear(64,nclass)
        self.nclass = nclass

    # def forward_mid(self,x):
    #     N,D,H,W = x.shape # 1,32,124,124
    #     Hf = self.ksize
    #     Wf = self.ksize
    #     # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
    #     X_col = im2col_indices(x, Hf, Wf, padding=self.pad, stride=self.stride, dilation=self.dilation)
    #     # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
    #     W_col = self.Wt.view(self.outchannel, -1)

    #     W_col_brk = W_col.view(self.outchannel, self.inchannel,-1).unsqueeze(3) # 64,64,9,1
    #     X_col_brk = X_col.view(self.inchannel,-1,X_col.shape[-1]).unsqueeze(0) # 1,64,9,28800
    #     midres = (X_col_brk*W_col_brk).sum(1) + self.bias.unsqueeze(1) # 64, 9, 28800
        
    #     out = midres.view(self.outchannel, Hf*Wf, H, W, N)
    #     out = out.permute(4, 2, 3, 1, 0).contiguous() # 1,124,124,9,64

    #     bound = (int(self.ksize/2))*self.dilation

    #     y_offset = torch.from_numpy(np.repeat(np.arange(-bound,bound+1,self.dilation), self.ksize)).type(device.FloatTensor)
    #     x_offset = torch.from_numpy(np.tile(np.arange(-bound,bound+1,self.dilation), self.ksize)).type(device.FloatTensor)
        
    #     return out, y_offset, x_offset


    def forward(self,x): # NxDxHxW
        # N,D,H,W = x.shape # 8,32,124,124
        # Hf = self.ksize
        # Wf = self.ksize
        # # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
        # X_col = im2col_indices(x, Hf, Wf, padding=self.pad, stride=self.stride, dilation=self.dilation)
        # # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
        # W_col = self.Wt.view(self.outchannel, -1)

        # # 20x9 x 9x500 = 20x500
        # out = W_col @ X_col + self.bias

        # # Reshape back from 20x500 to 5x20x10x10
        # # i.e. for each of our 5 images, we have 20 results with size of 10x10
        # out = out.view(self.outchannel, H, W, N)
        # out = out.permute(3, 0, 1, 2)
        # return out

        N,D,H,W = x.shape # 1,32,124,124
        Hf = self.ksize
        Wf = self.ksize
        # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
        X_col = im2col_indices(x, Hf, Wf, padding=self.pad, stride=self.stride, dilation=self.dilation)
        # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
        W_col = self.Wt.view(self.outchannel, -1)

        W_col_brk = W_col.view(self.outchannel, self.inchannel,-1).unsqueeze(3) # 64,64,9,1
        X_col_brk = X_col.view(self.inchannel,-1,X_col.shape[-1]).unsqueeze(0) # 1,64,9,28800
        midres = (X_col_brk*W_col_brk).sum(1) + self.bias.unsqueeze(1) # 64, 9, 28800
        
        feats = midres.view(self.outchannel, Hf*Wf, H, W, N)
        self.feats = feats.permute(4, 2, 3, 1, 0).contiguous() # 1,124,124,9,64

        bound = (int(self.ksize/2))*self.dilation

        yofs = torch.from_numpy(np.repeat(np.arange(-bound,bound+1,self.dilation), self.ksize)).type(device.FloatTensor)
        xofs = torch.from_numpy(np.tile(np.arange(-bound,bound+1,self.dilation), self.ksize)).type(device.FloatTensor)
        
        cmap = self.linear_final(self.feats) # 4,124,124,9,2
        # choose right cmap by clss
        # cmap2show = cmap[...,clss]
        # calculate drift measure
        y_drift = (torch.abs(cmap)*yofs[None,None,None,:,None]).sum(3)/torch.abs(cmap).sum(3) # broadcast # 4,124,124,2
        x_drift = (torch.abs(cmap)*xofs[None,None,None,:,None]).sum(3)/torch.abs(cmap).sum(3) # broadcast
        # drift heat map
        self.drift_map = torch.sqrt(x_drift**2+y_drift**2) # 4,124,124,2

        cmap_mod = cmap.sum(3) * torch.exp(-0.5*self.drift_map)

        return cmap_mod

        


