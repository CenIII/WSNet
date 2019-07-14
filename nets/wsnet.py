"""
high level support for doing this and that.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import Gap, KQ, Bayes

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class WeaklySupNet(nn.Module):
    """
    An end-to-end semantic segmentation net via cross-entropy loss only. 
    """
    def __init__(self,nclass):
        super(WeaklySupNet,self).__init__()     
        kq_dim = 16
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.kq = KQ(64,kq_dim)
        self.feature = nn.Sequential(
            nn.Conv2d(64, 64, (3,3),padding=1),
            nn.ReLU()
        )
        self.bayes = Bayes(64,kq_dim,2,n_heads=1,dif_pattern=[(3,6)],rel_pattern=[(5,3)])
        self.gap = Gap(64,nclass)
        self.nclass = nclass

    def getHeatmaps(self,classid):
        zzz = classid[:,None,None,None].repeat(1,self.gap.heatmaps.shape[1],self.gap.heatmaps.shape[2],1)
        hm = torch.gather(self.gap.heatmaps,3,zzz).squeeze()
        return hm

    def forward(self,x,label):
        bb = self.backbone(x)
        K,Q = self.kq(bb)
        feats = self.feature(bb)
        feats1, preds1 = self.bayes(feats,K,Q,label)
        pred = self.gap(feats1,save_hm=True)
        return preds1, pred
    