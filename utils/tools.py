import torch
import torch.nn.functional as F
import numpy as np
if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device

def multilabel_soft_pull_loss(input, target, weight=None,reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor
    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    # if size_average is not None or reduce is not None:
    #     reduction = _Reduction.legacy_get_string(size_average, reduce)

    loss = -(target * torch.log(torch.sigmoid((input)))) #+ (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values

    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        ret = loss.mean()
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret

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

def im2col_boundary(x, field_height, field_width, padding=1, stride=1, dilate=1,dist_to_center=None):
    # For boundary map, D=1.
    # dist_to_center [ksize,ksize]
    cols_list = [] 
    cols_max_list = []
    
    for dilate_i in range(1,dilate+1):
        padding = int(field_height/2)*dilate_i # assume the height& width is always the same
        p = padding
        x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)
        # import pdb;pdb.set_trace()
        k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                    stride, dilate_i)
        cols_i = x_padded[:, k, i, j] #torch.Size([B, fW*fH*D, W*H])
        cols_list.append(cols_i)
    cols_all = torch.stack(cols_list) # torch.Size([Dilate, B, fW*fH*D, W*H])
    for dilate_i in range(1,dilate+1):
        cols,_ = torch.max(cols_all[:dilate_i,:,:,:],dim=0) # torch.Size([B, fW*fH*D, W*H])
        cols_max_list.append(cols)
    cols_max_all = torch.stack(cols_max_list) # torch.Size([Dilate, B, fW*fH*D, W*H])
    dist_to_center = dist_to_center.reshape(1,1,-1,1).expand(1,cols_max_all.shape[1],-1,cols_max_all.shape[3]).contiguous() # torch.Size([1, B, fW*fH*D, W*H])
    import pdb;pdb.set_trace()
    cols = torch.gather(cols_max_all,dim=0,index=dist_to_center).squeeze(0)
    print(cols.shape)
    C = x.shape[1] #D
    # [5,mid] is the original pixel for (2,2)
    
    cols = cols.permute(1, 2, 0).contiguous().view(field_height * field_width * C, -1) # [fW*fH*D,W*H*B] (feature for each pixel ,num of pixels)
    print(cols.shape)
    return cols

def im2col_indices(x, field_height, field_width, padding=1, stride=1, dilate=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    # For boundary map, D=1
    # cols_list = []
    # for dilate_i in range(dilate):
        # padding = int(field_height/2)*dilate_i # assume the height& width is always the same
    p = padding
    x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                stride, dilate)
    cols = x_padded[:, k, i, j] #torch.Size([B, fW*fH*D, W*H])
    #     cols_list.append(cols)
    # cols_all = torch.stack(cols_list) ##torch.Size([Dilate, B, fW*fH*D, W*H])

    C = x.shape[1] #D
    # [5,mid] is the original pixel for (2,2)
    cols = cols.permute(1, 2, 0).contiguous().view(field_height * field_width * C, -1) # [fW*fH*D,W*H*B] (feature for each pixel ,num of pixels)
    
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