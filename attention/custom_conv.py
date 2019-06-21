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

    def forward(self,x): # NxDxHxW
        N,D,H,W = x.shape # 8,32,124,124
        Hf = self.ksize
        Wf = self.ksize
        


        K_trans = im2col_indices(K,Hf,Wf,padding=self.pad, stride=self.stride, dilate=self.dilate) # (3200,38440)
        Q_trans = Q.permute(1,2,3,0).contiguous().view(D,-1) # (128,38440)
        tmp = (K_trans.view(D,-1,K_trans.shape[-1])*Q_trans.unsqueeze(1)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,38440)
        att = torch.softmax(tmp,2) # (4,32,25,38440)
        V_trans = im2col_indices(V,Hf,Wf,padding=self.pad, stride=self.stride, dilate=self.dilate).view(self.nheads,self.hdim,Hf*Wf,-1)
        out = (V_trans*att).sum(2).view(D,H,W,N).permute(3,0,1,2)
        out = self.transform(out)
        return out