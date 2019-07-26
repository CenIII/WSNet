import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import im2col_indices, col2im_indices,im2col_boundary
if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device
import torch.nn.functional as F

class Gap(nn.Module):
    def __init__(self, in_channels, n_class):
        super(Gap, self).__init__()
        self.lin = nn.Conv2d(in_channels,n_class-1,1,bias=False) #nn.Linear(in_channels, n_class, bias=False)
        self.n_class = n_class

    def forward(self, x):
        N = x.shape[0]
        cam = self.lin(x) #.permute(0, 2, 3, 1)
        cam_2 = torch.sum(F.relu(cam),dim=1,keepdim=True)/2
        thres = 0.1*torch.max(cam_2.view(N,-1),dim=1)[0]
        cam_2 = thres[:,None,None,None]-cam_2#F.relu()+1e-5
        cam = torch.cat((cam,cam_2),dim=1)

        pred = torch.mean(cam.view(N, self.n_class, -1), dim=2)
        return pred, F.relu(cam) #F.relu(cam)+1. #F.leaky_relu(cam)

    def infer_class_maps(self, x):
        x = self.lin(x.permute(0, 2, 3, 1))
        return x

    def make_mask(self, label):
        N, W, H, _ = self.heatmaps.shape
        soft_mask = F.relu(self.heatmaps)
        mask = torch.gather(soft_mask, 3, label[:, None, None, None].repeat(1, W, H, 1)).squeeze().detach() # (8, 81, 81)
        mask_norm = (mask/torch.clamp(torch.max(mask.view(N, -1), dim=1)[0], 1.)[:, None, None]).unsqueeze(1)
        mask_norm[mask_norm>0.15]=1
        mask_norm[mask_norm<=0.15]=0
        return mask_norm

class LeakyLogGap(Gap):
    def __init__(self, in_channels, n_class):
        super(LeakyLogGap, self).__init__(in_channels, n_class)
    def forward(self, x, save_hm=False):
        N = x.shape[0]
        pre_hm = self.lin(x.permute(0, 2, 3, 1))
        x = torch.log(1 + F.relu(pre_hm))
        if save_hm:
            self.heatmaps = x
        x = torch.mean((x - 0.12 * F.relu(-pre_hm)).view(N, -1, self.n_class), dim=1)
        return x

class Boundary(nn.Module):
    def __init__(self,in_channels):
        super(Boundary,self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(3,3)),#,padding=1),
            nn.GroupNorm(4,in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,(1,1),padding=1),
            nn.GroupNorm(4, in_channels),
            nn.ReLU(),
            # nn.Conv2d(in_channels,in_channels,(3,3),padding=1),
            # nn.GroupNorm(8, in_channels),
            # nn.ReLU(),
            # nn.Conv2d(in_channels,in_channels,(3,3),padding=1),
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,1,(1,1)),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.nn(x)

class KQ(nn.Module):
    def __init__(self, in_channels, kq_dim):
        super(KQ, self).__init__()
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(in_channels, kq_dim, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(kq_dim, kq_dim, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(kq_dim, kq_dim, (1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(kq_dim, kq_dim, (1, 1)), 
            )
        self.q = nn.Sequential(
            nn.Conv2d(in_channels, kq_dim, (1, 1)), 
            )
        
    def forward(self, x):
        K = self.k(x)
        Q = self.q(x)
        return K, Q

    
class Infusion(nn.Module):
    def __init__(self, v_dim, kq_dim, n_heads=1, kernel_size=5, dilation=3):
        super(Infusion, self).__init__()  
        assert(kq_dim % n_heads == 0)
        self.v_dim = v_dim
        self.kq_dim = kq_dim
        self.n_heads = n_heads
        self.h_dim = int(kq_dim / n_heads)

    def forward(self,V,boundary,ksize,dilation,dist_to_center):
        # get a [1,1,fW*fH,W*H*B] attention 
        
        N, D, H, W = V.shape
        padding = int(ksize/2)*dilation
        Hf = ksize
        Wf = ksize
        # import pdb;pdb.set_trace()
        boundary_max = im2col_boundary(boundary, Hf, Wf, padding, 1, dilation,dist_to_center)
        boundary_sim = torch.ones_like(boundary_max)-boundary_max+1e-5
        boundary_sim = boundary_sim.unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 25, 52488])
        
        att =  boundary_sim/torch.sum(boundary_sim,dim=2,keepdim=True) #torch.softmax(boundary_sim, 2) # [1,1,fW*fH,W*H*B]
        att = att*((1-boundary).permute(1,2,3,0).contiguous().view(-1)[None,None,None,:])
        V_trans = im2col_indices(V, Hf, Wf, padding, 1, dilation).view(1, self.v_dim, Hf*Wf, -1) # torch.Size([1, 2, 9, 52488])
        out = (V_trans * att).sum(2).sum(0).view(D, H, W, N).permute(3, 0, 1, 2)/self.n_heads # torch.Size([8, 2, 81, 81])
        return out

    def forward_old(self, V, K, Q, ksize, dilation): # NxDxHxW
        # import pdb;pdb.set_trace()
        N, D, H, W = V.shape # torch.Size([8, 2, 81, 81])
        padding = int(ksize/2)*dilation #3
        Hf = ksize # kernal size 
        Wf = ksize
        K_trans = im2col_indices(K, Hf, Wf, padding, 1, dilation) # torch.Size([144, 52488]) [fW*fH*D,W*H*B]
        Q_trans = Q.permute(1, 2, 3, 0).contiguous().view(self.n_heads, self.h_dim, -1) # torch.Size([1, 16, 52488]) [1,D,W*H*B]
        tmp = (K_trans.view(self.n_heads, self.h_dim, -1, K_trans.shape[-1]) * Q_trans.unsqueeze(2)).view(self.n_heads, self.h_dim, Hf*Wf, -1) #torch.Size([1, 16, 9, 52488])
        tmp = tmp.sum(1, True)/np.sqrt(self.h_dim) # torch.Size([1, 1, 9, 52488])
        att = torch.softmax(tmp, 2) # torch.Size([1, 1, 9, 52488]) [1,1,fW*fH,W*H*B]
        V_trans = im2col_indices(V, Hf, Wf, padding, 1, dilation).view(1, self.v_dim, Hf*Wf, -1) # torch.Size([1, 2, 9, 52488])
        out = (V_trans * att).sum(2).sum(0).view(D, H, W, N).permute(3, 0, 1, 2)/self.n_heads # torch.Size([8, 2, 81, 81])
        return out

class Diffusion(nn.Module):
    def __init__(self, v_dim, kq_dim, n_heads=1):
        super(Diffusion, self).__init__()  
        assert(kq_dim % n_heads == 0)
        self.v_dim = v_dim
        self.kq_dim = kq_dim
        self.n_heads = n_heads
        self.h_dim = int(kq_dim/n_heads)

    def get_att_map(self):
        att_cols=self.att.view(Hf*Wf, -1).detach()
        att_new=col2im_indices(att_cols, [N, self.n_heads, H, W], Hf, Wf, padding, 1, dilation)
        att_new = torch.clamp(att_new, 2.)
        return att_new

    def forward(self, V, K, Q, ksize, dilation, mask):
        N, D, H, W = Q.shape # 8, 32, 124, 124
        Dv = V.shape[1]
        Hf = ksize
        Wf = ksize
        padding = int(ksize/2)*dilation
        
        K_trans = im2col_indices(K, Hf, Wf, padding, 1, dilation) # (800, 123008)
        Q_trans = Q.permute(1, 2, 3, 0).contiguous().view(D, -1) # (32, 123008)
        tmp = (K_trans.view(D, -1, K_trans.shape[-1]) * Q_trans.unsqueeze(1)).view(self.n_heads, self.h_dim, Hf*Wf, -1)
        tmp = tmp.sum(1, True)/np.sqrt(self.h_dim) # (4, 1, 5*5, 123008)
        self.att = torch.softmax(tmp, 2) # (4, 1, 25, 123008)
        mask = mask.squeeze().permute(1, 2, 0).contiguous().view(-1)
        self.att = self.att * mask[None, None, None, :] 

        V_trans = V.permute(1, 2, 3, 0).contiguous().view(self.n_heads, int(Dv/self.n_heads), 1, H*W*N) # (4, h_dim, 1, 123008)
        V_cols = (V_trans * self.att).view(Dv*Hf*Wf, -1) # (n_heads*h_dim*25, 123008)
        V_new = col2im_indices(V_cols, V.shape, Hf, Wf, padding, 1, dilation)
        return V_new

class Relation(nn.Module):
    def calculate_dist_pattern(self,rel_pattern):
        use_cuda = torch.cuda.is_available()
        pattern_dict = {}
        for ksize, _ in rel_pattern:
            pattern = torch.zeros([ksize,ksize])
            mid = int(ksize/2)
            for i in range(ksize):
                for j in range(ksize):
                    pattern[i][j]= max(abs(i-mid),abs(j-mid))
            pattern = pattern.long()
            if use_cuda:
                pattern=pattern.cuda()
            pattern_dict[ksize]=pattern
        return pattern_dict
        
    def __init__(self, in_channels, kq_dim, n_class, n_heads=1, rel_pattern=[(5, 3)]):
        super(Relation, self).__init__()
        self.rel_pattern = rel_pattern
        self.infuse = Infusion(in_channels, kq_dim, n_heads=n_heads)
        self.n_class = n_class
        self.dist_pattern_dict = self.calculate_dist_pattern(rel_pattern)

    def forward(self, feats, boundary):
        N = feats.shape[0]
        feats_r = []
        # import pdb;pdb.set_trace()
        for ksize, dilation in self.rel_pattern:
            dist_to_center = self.dist_pattern_dict[ksize]
            feats_r.append(self.infuse(feats,boundary, ksize, dilation,self.dist_pattern_dict[ksize]))
            
        feats_r = torch.stack(feats_r, dim=0).sum(0)
        pred_r = torch.mean(feats_r.view(N, self.n_class, -1), dim=2)
        return pred_r, feats_r