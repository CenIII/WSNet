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
        self.relation = Relation(64,16,1)
        self.feature = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3),padding=1),
            nn.ReLU()
            ) 
        self.cmap0 = nn.Linear(64,nclass)
        self.transform = nn.Conv2d(64,64,(3,3),padding=1)
        self.cmap1 = nn.Linear(64,nclass,bias=False)
        self.cmap2 = nn.Linear(64,nclass)
        self.tmasknet = nn.Linear(64,1)
        self.nclass = nclass

    def getAttention(self,classid):
        zzz = classid[:,None,None,None].repeat(1,self.heatmaps.shape[1],self.heatmaps.shape[2],1)
        hm = torch.gather(self.heatmaps,3,zzz).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        return hm

    def getRelHm(self,classid):
        zzz = classid[:,None,None,None].repeat(1,self.heatmaps2.shape[1],self.heatmaps2.shape[2],1)
        hm = torch.gather(self.heatmaps2,3,zzz).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        return hm

    def normTMask(self,target_mask):
        target_mask = target_mask.squeeze() # 
        target_mask = target_mask/torch.clamp(torch.max(target_mask.view(target_mask.shape[0],-1),dim=1)[0],1.)[:,None,None]
        target_mask[target_mask<0.25] = 0.
        return target_mask

    def forward(self,x,label,norm_att=False):
        bb = self.backbone(x) #torch.Size([2, 16, 64, 64])
        feats0 = self.feature(bb)

        pre_hm0 = self.cmap0(feats0.permute(0,2,3,1))
        heatmaps0 = torch.log(1+F.relu(pre_hm0))
        pred0 = torch.mean((heatmaps0 - 0.12*F.relu(-pre_hm0)).view(x.shape[0],-1,self.nclass),dim=1).squeeze()

        K,Q = self.diffuse(bb)
        feats0_trans = self.diffuse.transfer(feats0,label,6,norm_att=norm_att) #norm_tmask,
        # feats0_trans += self.diffuse.transfer(feats0,heatmaps0,label,3,norm_att=norm_att)

        feats1 = self.transform(feats0_trans)#torch.cat((feats0,feats0_trans),dim=1)) #torch.cat((feats0,feats0_trans),dim=1)
        pre_hm1 = self.cmap1(feats1.permute(0,2,3,1))
        self.heatmaps = pre_hm1 #torch.log(1+F.relu(pre_hm1))
        pred1 = torch.mean((self.heatmaps).view(x.shape[0],-1,self.nclass),dim=1).squeeze()  # - 0.12*F.relu(-pre_hm1)
        
        feats0_rel = self.relation(feats0,Q,K)
        pre_hm2 = self.cmap2(feats0_rel.permute(0,2,3,1))
        self.heatmaps2 = pre_hm2 #torch.log(1+F.relu(pre_hm2))
        pred2 = torch.mean((self.heatmaps2).view(x.shape[0],-1,self.nclass),dim=1).squeeze() # - 0.12*F.relu(-pre_hm2)
        
        return pred0, pred1, pred2, K, self.heatmaps #pred_tmask,
    