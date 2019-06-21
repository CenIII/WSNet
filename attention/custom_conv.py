import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices

class CustmConv(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size=5,stride=1,dilate=1):
        super(Relation,self).__init__()  
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.ksize = kernel_size
        self.pad = int(kernel_size/2)*dilate
        self.dilate = dilate
        self.stride = stride
        self.Wt = torch.nn.Parameter(data=torch.zeros(outchannel,inchannel,self.ksize,self.ksize), requires_grad=True)
        self.Wt.data.uniform_(-1, 1)
        self.b = torch.nn.Parameter(data=torch.zeros(outchannel,1), requires_grad=True)

    def forward(self,x): # NxDxHxW
        N,D,H,W = x.shape # 8,32,124,124
        Hf = self.ksize
        Wf = self.ksize
        # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
        X_col = im2col_indices(x, Hf, Wf, padding=self.pad, stride=self.stride, dilate=self.dilate)
        # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
        W_col = self.Wt.reshape(self.outchannel, -1)

        # 20x9 x 9x500 = 20x500
        out = W_col @ X_col + self.b

        # Reshape back from 20x500 to 5x20x10x10
        # i.e. for each of our 5 images, we have 20 results with size of 10x10
        out = out.reshape(self.outchannel, H, W, self.outchannel)
        out = out.transpose(3, 0, 1, 2)

        return out