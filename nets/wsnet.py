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
        self.dilated = nn.ModuleList([nn.Sequential(
            nn.Conv2d(64, 32, (7, 7), dilation=3, padding=9), 
            nn.ReLU()
        ),
            nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), dilation=6, padding=6), 
            nn.ReLU()
        ),
            nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), dilation=3, padding=6), 
            nn.ReLU()
        ),
            nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), dilation=2, padding=4), 
            nn.ReLU()
        )
        ])

        self.gap0 = Gap(128, nclass)
        self.nclass = nclass

    def getHeatmaps(self, classid):
        zzz = classid[:, None, None, None].repeat(1, self.gap0.heatmaps.shape[1], self.gap0.heatmaps.shape[2], 1)
        hm = torch.gather(self.gap0.heatmaps, 3, zzz).squeeze()
        return hm
    
    def getHmRel(self,classid):
        zzz = classid[:, None, None, None].repeat(1, self.bayes.hm_rel.shape[1], self.bayes.hm_rel.shape[2], 1)
        hm = torch.gather(self.bayes.hm_rel, 3, zzz).squeeze()
        return hm
    def forward(self, x, label):
        bb = self.backbone(x)
        feats = []
        for i in range(4):
            feats.append(self.dilated[i](bb))
        feats = torch.cat(feats,dim=1)
        pred0 = self.gap0(feats, save_hm=True)
        return pred0
    