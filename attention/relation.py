import torch
import torch.nn as nn

class Relation(nn.Module):
    def __init__(self,inchannel,outchannel,nheads,kernel_size=4):
        super(Relation,self).__init__()  
        assert(outchannel%nheads==0)
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.nheads = nheads
        self.hdim = int(outchannel/nheads)
        self.ksize = kernel_size
        self.qkv = nn.Conv(inchannel,int(3*outchannel),(1,1)) # q k v
    
    def forward(self,x):
        qkv = self.qkv(x)
        Q = qkv[:,:,:,:self.outchannel]
        K = qkv[:,:,:,self.outchannel:(2*self.outchannel)]
        V = qkv[:,:,:,(2*self.outchannel):(3*self.outchannel)]
        

    