"""
high level support for doing this and that.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import Gap, KQ, Relation

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class WeaklySupNet(nn.Module):
    """
    An end-to-end semantic segmentation net via cross-entropy loss only. 
    """
    def __init__(self, nclass):
        super(WeaklySupNet, self).__init__()     
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
        self.kq = KQ(64, kq_dim)
        self.branch_local = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1)
        )
        self.branch_relation = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1))#, padding=1)
        )
        self.gap0 = Gap(64, nclass)
        self.relation = Relation(2, kq_dim, 2, n_heads=1, rel_pattern=[(3,3),(5,2),(3,5)])
        self.nclass = nclass

    def getHeatmaps(self, classid):
        zzz = classid[:, None, None, None].repeat(1, self.heatmaps.shape[1], self.heatmaps.shape[2], 1)
        hm = torch.gather(self.heatmaps, 3, zzz).squeeze()
        return hm
    
    def getInitHeatmaps(self, classid):
        zzz = classid[:, None, None, None].repeat(1, self.initheatmaps.shape[1], self.initheatmaps.shape[2], 1)
        hm = torch.gather(self.initheatmaps, 3, zzz).squeeze()
        return hm

    def forward(self, x, label):
        bb = self.backbone(x)
        feats_lc = self.branch_local(bb)
        # feats_rel = self.branch_relation(bb)
        
        K, Q = self.kq(bb)
        pred0, cam0 = self.gap0(feats_lc)
        pred1, cam1 = self.relation(cam0, K, Q)
        pred2, cam2 = self.relation(cam1, K, Q)

        self.initheatmaps = cam0.permute(0,2,3,1)
        self.heatmaps = cam2.permute(0,2,3,1)
    
        return [pred1, pred2], pred0 