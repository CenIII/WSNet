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
        # self.softmax2d = nn.Softmax2d()

    def forward(self,x): # NxDxHxW
        qkv = self.qkv(x)
        Q = qkv[:,:self.outchannel]
        K = qkv[:,self.outchannel:(2*self.outchannel)]
        V = qkv[:,(2*self.outchannel):(3*self.outchannel)]
        # local attention
        N,D,H,W = Q.shape
        ofs = int(self.ksize/2)
        out = []
        # for h in range(H):
        #     for w in range(W):
        #         ks = K[:,:,max(0,h-ofs):min(H,h+ofs+1),max(0,w-ofs):min(W,w+ofs+1)] # NxDxkxk
        #         vs = V[:,:,max(0,h-ofs):min(H,h+ofs+1),max(0,w-ofs):min(W,w+ofs+1)] # NxDxkxk
        #         _,_,kh,kw = ks.shape
        #         q = Q[:,:,h,w].unsqueeze(2).unsqueeze(2).repeat(1,1,kh,kw) # NxDxkxk

        #         # attention weights
        #         tmp = (ks*q).view(N,self.nheads,self.hdim,kh,kw).sum(2) # Nxnheadsxkxk
        #         att = torch.softmax(tmp.view(N,self.nheads,-1),dim=2).view_as(tmp).unsqueeze(2).repeat(1,1,self.hdim,1,1) # Nxnheadsxhdimxkxk
        #         res = vs.view(N,self.nheads,self.hdim,kh,kw)*att # Nxnheadsxhdimxkxk
        #         res = res.sum(3,True).sum(4,True).squeeze().view(N,self.outchannel) # Nxoutchannel
        #         out.append(res)
        # out = torch.stack(out,dim=2).view(N,-1,H,W) # NxDxHxW
        # Hf = self.ksize
        # Wf = self.ksize
        # K_trans = im2col_indices(K,Hf,Wf,2) # (3200,38440)
        # Q_trans = Q.repeat(1,Hf*Wf,1,1).permute(1,2,3,0).contiguous().view(D*Hf*Wf,-1) # (3200,38440)
        # tmp = (K_trans*Q_trans).view(self.nheads,self.hdim,Hf*Wf,-1) # (4,32,5*5,38440)
        # tmp = tmp.sum(1,True) # (4,1,5*5,38440)
        # att = torch.softmax(tmp,2).repeat(1,self.hdim,1,1) # (4,32,25,38440)
        # V_trans = im2col_indices(V,Hf,Wf,2).view(self.nheads,self.hdim,Hf*Wf,-1)
        # out = (V_trans*att).sum(2).view(D,H,W,N).permute(3,0,1,2)
        #
        Hf = self.ksize
        Wf = self.ksize
        K_trans = im2col_indices(K,Hf,Wf,2) # (3200,38440)
        Q_trans = Q.permute(1,2,3,0).contiguous().view(D,-1) # (128,38440)
        tmp = (K_trans.view(D,-1,K_trans.shape[-1])*Q_trans.unsqueeze(1)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,38440)
        att = torch.softmax(tmp,2) # (4,32,25,38440)
        V_trans = im2col_indices(V,Hf,Wf,2).view(self.nheads,self.hdim,Hf*Wf,-1)
        out = (V_trans*att).sum(2).view(D,H,W,N).permute(3,0,1,2)
        return out

    