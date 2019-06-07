import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.nn as nn

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class WeaklySupNet(nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self):
        super(WeaklySupNet,self).__init__()     

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),#, padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2)#,
            # nn.Conv2d(16, 8, (3, 3)),
            # nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.linear_final = nn.Linear(128,3)
        self.linear_mining = nn.Linear(128,3)

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
        return hm#torch.exp(hm)

    def getAttention_m(self,classid):
        zzz = classid[:,None,None,None].repeat(1,self.heatmaps_m.shape[1],self.heatmaps_m.shape[2],1)
        hm = torch.gather(self.heatmaps_m,3,zzz).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        return hm#torch.exp(hm)        

    def forward(self,x):
        feats = self.conv(x) #torch.Size([2, 16, 64, 64])
        self.feats = feats.permute(0,2,3,1)
        self.feats.register_hook(self.getHMgrad)

        pre_hm = self.linear_final(self.feats)
        self.heatmaps = torch.log(1+F.relu(pre_hm)) #- 0.2*F.relu(-pre_hm)#torch.sqrt() 
        
        # linear # softmax
        # pred = F.log_softmax(torch.mean(self.heatmaps.view(2,-1,2),dim=1).squeeze(),dim=1)
        pred = F.log_softmax(torch.mean((self.heatmaps - 0.12*F.relu(-pre_hm)).view(self.feats.shape[0],-1,2),dim=1).squeeze(),dim=1)

        # feature mining
        pre_hm_m = self.linear_mining(self.feats.detach())
        self.heatmaps_m = torch.log(1+F.relu(pre_hm_m))
        pred_m = F.sigmoid(torch.mean((self.heatmaps_m - 0.12*F.relu(-pre_hm_m)).view(self.feats.shape[0],-1,2),dim=1).squeeze(),dim=1)

        return pred, pred_m



# class Criterion(nn.Module):
#     """docstring for Criterion"""
#     def __init__(self):
#         super(Criterion, self).__init__()
    
#     def forward(self,pred,label):
#         loss = 0.
#         for i in range(len(pred)):
#             pr = pred[i][label[i]]
#             targ = pr.clone().detach()+2.
#             loss += (targ-pr)**2
#         return loss
        
