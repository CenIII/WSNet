import torch
import torch.nn as nn


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

class Relation(nn.Module):
    def __init__(self,inchannel,outchannel,nheads,kernel_size=5):
        super(Relation,self).__init__()  
        assert(outchannel%nheads==0)
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.nheads = nheads
        self.hdim = int(outchannel/nheads)
        self.ksize = kernel_size
        self.qkv = nn.Conv2d(inchannel,int(3*outchannel),(1,1)) # q k v
        # self.softmax2d = nn.Softmax2d()

    def forward(self,x): # NxDxHxW
        qkv = self.qkv(x)
        Q = qkv[:,:self.outchannel]
        K = qkv[:,self.outchannel:(2*self.outchannel)]
        V = qkv[:,(2*self.outchannel):(3*self.outchannel)]
        # local attention
        N,D,H,W = Q.shape
        ofs = int(self.ksize/2)
        out = []
        for h in range(H):
            for w in range(W):
                ks = K[:,:,max(0,h-ofs):min(H,h+ofs+1),max(0,w-ofs):min(W,w+ofs+1)] # NxDxkxk
                vs = V[:,:,max(0,h-ofs):min(H,h+ofs+1),max(0,w-ofs):min(W,w+ofs+1)] # NxDxkxk
                _,_,kh,kw = ks.shape
                q = Q[:,:,h,w].unsqueeze(2).unsqueeze(2).repeat(1,1,kh,kw) # NxDxkxk

                # attention weights
                tmp = (ks*q).view(N,self.nheads,self.hdim,kh,kw).sum(2) # Nxnheadsxkxk
                att = torch.softmax(tmp.view(N,self.nheads,-1),dim=2).view_as(tmp).unsqueeze(2).repeat(1,1,self.hdim,1,1) # Nxnheadsxhdimxkxk
                res = vs.view(N,self.nheads,self.hdim,kh,kw)*att # Nxnheadsxhdimxkxk
                res = res.sum(3,True).sum(4,True).squeeze().view(N,self.outchannel) # Nxoutchannel
                out.append(res)
        out = torch.stack(out,dim=2).view(N,-1,H,W) # NxDxHxW
        return out

    