import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices

class Relation(nn.Module):
    def __init__(self,inchannel,outchannel,nheads,kernel_size=5):
        super(Relation,self).__init__()  
        assert(outchannel%nheads==0)
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.nheads = nheads
        self.hdim = int(outchannel/nheads)
        self.ksize = kernel_size
        self.qkv = nn.Conv2d(inchannel,int(3*outchannel),(1,1),bias=False) # q k v
        self.transform = nn.Conv2d(outchannel,outchannel,(1,1),bias=False)

    def forward(self,x): # NxDxHxW
        qkv = self.qkv(x)
        Q = qkv[:,:self.outchannel]
        K = qkv[:,self.outchannel:(2*self.outchannel)]
        V = qkv[:,(2*self.outchannel):(3*self.outchannel)]
        # local attention
        N,D,H,W = Q.shape
        Hf = self.ksize
        Wf = self.ksize
        K_trans = im2col_indices(K,Hf,Wf,2,1,1) # (3200,38440)
        Q_trans = Q.permute(1,2,3,0).contiguous().view(D,-1) # (128,38440)
        tmp = (K_trans.view(D,-1,K_trans.shape[-1])*Q_trans.unsqueeze(1)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,38440)
        att = torch.softmax(tmp,2) # (4,32,25,38440)
        V_trans = im2col_indices(V,Hf,Wf,2,1,1).view(self.nheads,self.hdim,Hf*Wf,-1)
        out = (V_trans*att).sum(2).view(D,H,W,N).permute(3,0,1,2)
        out = self.transform(out)
        return out

    