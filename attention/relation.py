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
    def __init__(self,inchannel,outchannel,nheads=1,kernel_size=5,dilation=3):
        super(Relation,self).__init__()  
        assert(outchannel%nheads==0)
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.nheads = nheads
        self.hdim = int(outchannel/nheads)
        self.ksize = kernel_size
        self.dilation = dilation
        self.padding = int(self.ksize/2)*dilation

    def forward(self,V,K,Q): # NxDxHxW
        N,D,H,W = V.shape # 8,32,124,124
        Hf = self.ksize
        Wf = self.ksize
        K_trans = im2col_indices(K,Hf,Wf,self.padding,1,self.dilation) # (3200,38440)
        Q_trans = Q.permute(1,2,3,0).contiguous().view(self.nheads,self.hdim,-1) # (128,38440)
        tmp = (K_trans.view(self.nheads,self.hdim,-1,K_trans.shape[-1])*Q_trans.unsqueeze(2)).view(self.nheads,self.hdim,Hf*Wf,-1)
        tmp = tmp.sum(1,True) # (4,1,5*5,38440)
        att = torch.softmax(tmp,2) # (4,1,25,38440)
        V_trans = im2col_indices(V,Hf,Wf,self.padding,1,self.dilation).view(1,self.inchannel,Hf*Wf,-1)
        out = (V_trans*att).sum(2).sum(0).view(D,H,W,N).permute(3,0,1,2)/self.nheads
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
            nn.Conv2d(inchannel, inchannel, (1,1)),#,padding=1),#, padding=(1,0)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel,(1,1)),#,padding=1) # q k v
            )
        self.q = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, (1,1)),#,padding=1),#, padding=(1,0)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel,(1,1)),#,padding=1)#6,dilation=dilation) # q k v
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel,(1,1)),
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

    def get_att_map(self):
        # check self att max value 
        att_cols=self.att.view(Hf*Wf,-1).detach()
        att_new=col2im_indices(att_cols,[N,self.nheads,H,W],Hf,Wf,padding,1,dilation)
        att_new = torch.clamp(att_new,2.)
        return att_new

    # transfer get v and gen new v
    def transfer(self,V,label,dilation,norm_att=False): #target_mask,
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
        
        V_trans = V.permute(1,2,3,0).contiguous().view(self.nheads,int(Dv/self.nheads),1,H*W*N) # (4,hdim,1,123008)
        V_cols = (V_trans*self.att).view(Dv*Hf*Wf,-1) # (nheads*hdim*25,123008)
        V_new = col2im_indices(V_cols,V.shape,Hf,Wf,padding,1,dilation)
        return V_new

    # forward get KQ
    def forward(self,x): # NxDxHxW
        self.K = self.k(x)
        self.Q = self.q(x)
        return self.K, self.Q