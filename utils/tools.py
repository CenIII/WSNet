import torch
import torch.nn.functional as F
import numpy as np
if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1, dilate=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding / dilate - field_height) % stride == 0
    assert (W + 2 * padding / dilate - field_height) % stride == 0
    out_height = int((H + 2 * padding / dilate - field_height) / stride + 1)
    out_width = int((W + 2 * padding / dilate - field_width) / stride + 1)

    i0 = np.repeat(np.arange(0,field_height*dilate,dilate), field_width) #(9,)
    i0 = np.tile(i0, C) #(1152,)
    i1 = stride * np.repeat(np.arange(out_height), out_width) #(3844,)
    j0 = np.tile(np.arange(0,field_width*dilate,dilate), field_height * C)  #(1152,)
    j1 = stride * np.tile(np.arange(out_width), out_height) #(3844,)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1) #(1152, 3844)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1) #(1152, 3844)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1) #(1152, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1, dilate=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride, dilate)

    cols = x_padded[:, k, i, j] #torch.Size([8, 800, 15625])
    C = x.shape[1]
    cols = cols.permute(1, 2, 0).contiguous().view(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1, dilate=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = torch.zeros((N, C, H_padded, W_padded)).type(device.FloatTensor)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                stride,dilate)
    cols_reshaped = cols.view(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.permute(2, 0, 1) #[4, 576, 15376]
    # np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    ll = torch.from_numpy(np.arange(N))[:,None,None].repeat(1,C* field_height * field_width,H*W).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    kk = torch.from_numpy(k)[None,:,:].repeat(N,1,H*W).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    ii = torch.from_numpy(i)[None,:,:].repeat(N,1,1).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    jj = torch.from_numpy(j)[None,:,:].repeat(N,1,1).view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]
    cols_sliced = cols_reshaped.view(N,C,-1,H*W).permute(2,0,1,3).contiguous().view(field_height * field_width,-1) # [4, 64, 9, 15376]

    for pt in range(field_height * field_width):
        x_padded[ll[pt],kk[pt],ii[pt],jj[pt]] += cols_sliced[pt]
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]