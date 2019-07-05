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
from .relation import Relation, Diffusion

class WeaklySupNet(nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,nclass):
        super(WeaklySupNet,self).__init__()     

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5)),#, padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.diffuse = Diffusion(64,16,1)
        self.feature = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3),padding=1),
            nn.ReLU()
            ) 
        self.cmap0 = nn.Linear(64,nclass)
        self.transform = nn.Conv2d(128,64,(3,3),padding=1)
        self.cmap1 = nn.Linear(64,nclass)
        self.tmasknet = nn.Linear(64,1)
        self.nclass = nclass

    def getGradAttention(self):
        hm = torch.abs(self.hmgrad[:,:,:,0])
        return hm
    def getAttention(self,classid):
        zzz = classid[:,None,None,None].repeat(1,self.heatmaps.shape[1],self.heatmaps.shape[2],1)
        hm = torch.gather(self.heatmaps,3,zzz).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        return hm

    def getTMask(self):
        return self.tmask.squeeze()

    def normTMask(self,target_mask):
        target_mask = target_mask.squeeze() # 
        target_mask = target_mask/torch.clamp(torch.max(target_mask.view(target_mask.shape[0],-1),dim=1)[0],1.)[:,None,None]
        target_mask[target_mask<0.25] = 0.
        return target_mask

    def forward(self,x,label):
        bb = self.backbone(x) #torch.Size([2, 16, 64, 64])
        feats0 = self.feature(bb)

        pre_hm0 = self.cmap0(feats0.permute(0,2,3,1))
        heatmaps0 = torch.log(1+F.relu(pre_hm0))
        pred0 = torch.mean((heatmaps0 - 0.12*F.relu(-pre_hm0)).view(x.shape[0],-1,self.nclass),dim=1).squeeze()

        # target mask 
        pre_tmask = self.tmasknet(feats0.detach().permute(0,2,3,1))
        tmask = torch.log(1+F.relu(pre_tmask))
        norm_tmask = self.normTMask(tmask)
        self.tmask = norm_tmask
        pred_tmask = torch.mean((tmask - 0.12*F.relu(-pre_tmask)).view(x.shape[0],-1,1),dim=1)

        K,Q = self.diffuse(bb)
        feats0_trans = self.diffuse.transfer(feats0,heatmaps0,norm_tmask,label,6)
        feats0_trans += self.diffuse.transfer(feats0,heatmaps0,norm_tmask,label,3)

        feats1 = self.transform(torch.cat((feats0,feats0_trans),dim=1)) #torch.cat((feats0,feats0_trans),dim=1)
        pre_hm1 = self.cmap1(feats1.permute(0,2,3,1))
        self.heatmaps = torch.log(1+F.relu(pre_hm1))
        pred1 = torch.mean((self.heatmaps - 0.12*F.relu(-pre_hm1)).view(x.shape[0],-1,self.nclass),dim=1).squeeze()

        return pred0, pred1, pred_tmask, K, self.heatmaps
        
