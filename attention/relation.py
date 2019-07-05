import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices, col2im_indices
if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device
import torch.nn.functional as F

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
        Q_trans = Q.permute(1,2,3,0).contiguous().view(self.nheads,self.hdim,-1) # (128,38440)
        tmp = (F.normalize(K_trans.view(self.nheads,self.hdim,-1,K_trans.shape[-1]),dim=1)*F.normalize(Q_trans,dim=1).unsqueeze(2)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,38440)
        att = torch.softmax(10*tmp,2) # (4,32,25,38440)
        V_trans = im2col_indices(V,Hf,Wf,2,1,1).view(self.nheads,self.hdim,Hf*Wf,-1)
        out = (V_trans*att).sum(2).view(D,H,W,N).permute(3,0,1,2)
        out = self.transform(out)
        return out

class Diffusion(nn.Module):
    def __init__(self,inchannel,outchannel,nheads=1,kernel_size=3,dilation=6):
        super(Diffusion,self).__init__()  
        assert(outchannel%nheads==0)
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.nheads = nheads
        self.hdim = int(outchannel/nheads)
        self.ksize = kernel_size
        self.dilation = dilation
        self.padding = int(self.ksize/2)*dilation
        self.k = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, (3,3),padding=1),#, padding=(1,0)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel,(3,3),padding=1) # q k v
            )
        self.q = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, (3,3),padding=1),#, padding=(1,0)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel,(3,3),padding=1)#6,dilation=dilation) # q k v
            )

    def drawDiffuseMap(self,n):
        N,D,H,W = self.Q.shape
        attmap = self.att[0].squeeze().view(self.ksize*self.ksize,H,W,N)[...,n] #(9,124,124)
        bound = (int(self.ksize/2))*self.dilation

        yofs = torch.from_numpy(np.repeat(np.arange(-bound,bound+1,self.dilation), self.ksize)).type(device.FloatTensor)
        xofs = torch.from_numpy(np.tile(np.arange(-bound,bound+1,self.dilation), self.ksize)).type(device.FloatTensor)
        
        # calculate drift measure
        y_drift = (torch.abs(attmap)*yofs[:,None,None]).sum(0)#/torch.abs(attmap).sum(2) # broadcast
        x_drift = (torch.abs(attmap)*xofs[:,None,None]).sum(0)#/torch.abs(attmap).sum(2) # broadcast
        # plot drift heat map
        drift_map = torch.sqrt(x_drift**2+y_drift**2)

        # plt.figure(2)
        # plt.imshow(drift_map.data.cpu().numpy())
        # plt.show()
        return drift_map

    # transfer get v and gen new v
    def transfer(self,V,soft_mask,target_mask,label,dilation):
        # local attention
        N,D,H,W = self.Q.shape # 8,32,124,124
        Dv = V.shape[1]
        Hf = self.ksize
        Wf = self.ksize
        padding = int(self.ksize/2)*dilation
        
        K_trans = im2col_indices(self.K,Hf,Wf,padding,1,dilation) # (800,123008)
        Q_trans = self.Q.permute(1,2,3,0).contiguous().view(D,-1) # (32,123008)
        tmp = (K_trans.view(D,-1,K_trans.shape[-1])*Q_trans.unsqueeze(1)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,123008)
        self.att = torch.softmax(tmp,2) # (4,1,25,123008)
        # todo: use soft mask to att
        mask = torch.gather(soft_mask,3,label[:,None,None,None].repeat(1,soft_mask.shape[1],soft_mask.shape[2],1)).squeeze().detach() # (8,81,81)
        mask_norm = mask/torch.clamp(torch.max(mask.view(N,-1),dim=1)[0],1.)[:,None,None]

        mask = mask_norm.permute(1,2,0).contiguous().view(-1)
        self.att = self.att*mask[None,None,None,:] 

        V_trans = V.permute(1,2,3,0).contiguous().view(self.nheads,int(Dv/self.nheads),1,H*W*N) # (4,hdim,1,123008)
        # how to add back?
        V_cols = (V_trans*self.att).view(Dv*Hf*Wf,-1) # (nheads*hdim*25,123008)
        V_new = col2im_indices(V_cols,V.shape,Hf,Wf,padding,1,dilation)
        V_new = V_new*target_mask.unsqueeze(1)
        return V_new

    # forward get KQ
    def forward(self,x): # NxDxHxW
        self.K = self.k(x)
        self.Q = self.q(x)
        return self.Q, self.K