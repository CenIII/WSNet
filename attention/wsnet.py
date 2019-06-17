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
class WeaklySupNet(nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,nclass):
        super(WeaklySupNet,self).__init__()     

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),#, padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, (3, 3))
            
            )
        self.rel = Relation(64,64,4)
        self.linear_final = nn.Linear(128,nclass)
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

    def forward(self,x):
        feats = self.conv(x) #torch.Size([2, 16, 64, 64])
        rel_feats = self.rel(feats)
        feats = torch.cat((feats,rel_feats),dim=1)
        self.feats = feats.permute(0,2,3,1)
        self.feats.register_hook(self.getHMgrad)

        pre_hm = self.linear_final(self.feats)
        self.heatmaps = torch.log(1+F.relu(pre_hm)) #- 0.2*F.relu(-pre_hm)#torch.sqrt() 
        
        # linear # softmax
        pred = torch.mean((self.heatmaps - 0.12*F.relu(-pre_hm)).view(self.feats.shape[0],-1,self.nclass),dim=1).squeeze()
        return pred
        
