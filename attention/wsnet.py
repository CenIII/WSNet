"""
high level support for doing this and that.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from .relation import Relation
from .custom_conv import CustmConv, CalibConv
import matplotlib.pyplot as plt 

class WeaklySupNet(nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,nclass):
        super(WeaklySupNet,self).__init__()     

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
            )
        self.calib_conv = CalibConv(64, 64, 3, dilation=5,nclass=nclass)
        # self.linear_final = nn.Linear(64,nclass)
        self.nclass = nclass

    def getHMgrad(self,grad):
        self.hmgrad = grad
        return
    def getGradAttention(self):
        # zzz = classid[:,None,None,None].repeat(1,self.hmgrad.shape[1],self.hmgrad.shape[2],1)
        # hm = torch.gather(self.hmgrad,3,zzz).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        hm = torch.abs(self.hmgrad[:,:,:,0])
        return hm
    def getAttention(self,classid):
        zzz = classid[:,None,None,None].repeat(1,self.heatmaps.shape[1],self.heatmaps.shape[2],1)
        hm = torch.gather(self.heatmaps,3,zzz).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        return hm

    def drawDriftHeatMap(self,x,clss):
        feats = self.conv(x)
        hm = self.calib_conv(feats) # 1,124,124,9,64
        drift_map = self.calib_conv.drift_map
        plt.imshow(drift_map[...,clss].data.cpu().numpy())
        plt.show()

    def forward(self,x):
        feats = self.conv(x) #torch.Size([2, 16, 64, 64])
        pre_hm = self.calib_conv(feats) # 4,124,124,2
        # self.calib_conv.feats.register_hook(self.getHMgrad)
        self.heatmaps = pre_hm#F.leaky_relu(pre_hm)#torch.log(1+F.relu(pre_hm)) #- 0.2*F.relu(-pre_hm)#torch.sqrt() 
        # linear # softmax
        pred = torch.mean((self.heatmaps).view(x.shape[0],-1,self.nclass),dim=1).squeeze() #- 0.12*F.relu(-pre_hm)
        return pred
        
