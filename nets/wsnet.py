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
        self.kq = KQ(64, kq_dim)
        self.feature = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1), 
            nn.ReLU()
        )
        self.gap0 = Gap(64, nclass)
        self.bayes = Bayes(2, kq_dim, 2, n_heads=1, dif_pattern=[(3, 6),(3, 3),(5,3),(5,5)], rel_pattern=[(7, 3),(3,3),(5,1),(5,3),(5,5)])#
        # self.gap = Gap(64, nclass)
        self.nclass = nclass

    def getHeatmaps(self, classid):
        zzz = classid[:, None, None, None].repeat(1, self.bayes.heatmaps.shape[1], self.bayes.heatmaps.shape[2], 1)
        hm = torch.gather(self.bayes.heatmaps, 3, zzz).squeeze()
        return hm
    
    def getMask(self):
        return self.mask
    
    def getHmRel(self,classid):
        zzz = classid[:, None, None, None].repeat(1, self.bayes.hm_rel.shape[1], self.bayes.hm_rel.shape[2], 1)
        hm = torch.gather(self.bayes.hm_rel, 3, zzz).squeeze()
        return hm
    def forward(self, x, label):
        bb = self.backbone(x)
        K, Q = self.kq(bb)
        feats = self.feature(bb)
        pred0 = self.gap0(feats, save_hm=True)
        self.mask = self.gap0.make_mask(label)

        feats1, preds1 = self.bayes(self.gap0.heatmaps,self.mask, K, Q)
        # pred = torch.mean(feats1.view(feats1.shape[0], 2, -1), dim=2)#self.gap(feats1, save_hm=True)
        # preds1.append(pred)
        return preds1, pred0
    