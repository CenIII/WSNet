"""
high level support for doing this and that.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import Gap, KQ, Relation,Boundary

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
        self.backbone = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), padding=2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(3), 
        ), nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        ), nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
            )
        ])
        self.parymid = nn.MaxPool2d(2)
        
        self.boundary = Boundary(160)
        self.branch_local = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
        )
        
        self.gap0 = Gap(64, nclass)
        self.relation = Relation(nclass, kq_dim, nclass, n_heads=1, rel_pattern=[(3,3),(3,1),(3,5),(5,1)])
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
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        bb = self.backbone[2](x2)
        feats_rel = torch.cat((self.parymid(x1),x2,bb),dim=1)
        feats_lc = self.branch_local(bb)
        
        boundary = self.boundary(feats_rel)
        pred0, cam0 = self.gap0(feats_lc)
        pred1, cam1 = self.relation(cam0, boundary)
        pred2, cam2 = self.relation(cam1, boundary)
        pred3, cam3 = self.relation(cam2, boundary)

        self.bmap = boundary.squeeze()
        self.initheatmaps = cam0.permute(0,2,3,1)
        self.heatmaps = cam3.permute(0,2,3,1)
    
        return [pred1, pred2, pred3], pred0 