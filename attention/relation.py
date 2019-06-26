import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices, col2im_indices

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
        N,D,H,W = Q.shape # 8,32,124,124
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

class Diffusion(nn.Module):
    def __init__(self,inchannel,outchannel,nheads=1,kernel_size=3,dilation=4):
        super(Diffusion,self).__init__()  
        assert(outchannel%nheads==0)
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.nheads = nheads
        self.hdim = int(outchannel/nheads)
        self.ksize = kernel_size
        self.dilation = dilation
        self.padding = int(self.ksize/2)*dilation
        self.qk = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, (3, 3)),#, padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(inchannel,int(2*outchannel),(3,3),bias=False) # q k v
            )

    # transfer get v and gen new v
    def transfer(self,V):
        # local attention
        N,D,H,W = self.Q.shape # 8,32,124,124
        Hf = self.ksize
        Wf = self.ksize
        K_trans = im2col_indices(self.K,Hf,Wf,self.padding,1,self.dilation) # (800,123008)
        Q_trans = self.Q.permute(1,2,3,0).contiguous().view(D,-1) # (32,123008)
        tmp = (K_trans.view(D,-1,K_trans.shape[-1])*Q_trans.unsqueeze(1)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,123008)
        att = torch.softmax(tmp,2) # (4,1,25,123008)
        V_trans = V.permute(1,2,3,0).contiguous().view(self.nheads,self.hdim,1,H*W*N) # (4,hdim,1,123008)
        # how to add back?
        V_cols = (V_trans*att).view(self.nheads*self.hdim*Hf*Wf,-1) # (nheads*hdim*25,123008)
        V_new = col2im_indices(V_cols,Hf,Wf,self.padding,1,self.dilation)
        return V_new

    # forward get KQ
    def forward(self,x): # NxDxHxW
        qk = self.qk(x)
        self.Q = qk[:,:self.outchannel]
        self.K = qk[:,self.outchannel:(2*self.outchannel)]
        return self.Q, self.K