import torch.nn.functional as F
import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width) #(9,)
    i0 = np.tile(i0, C) #(1152,)
    i1 = stride * np.repeat(np.arange(out_height), out_width) #(3844,)
    j0 = np.tile(np.arange(field_width), field_height * C)  #(1152,)
    j1 = stride * np.tile(np.arange(out_width), out_height) #(3844,)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1) #(1152, 3844)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1) #(1152, 3844)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1) #(1152, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = F.pad(x, (p, p, p, p), mode='constant',value=0)

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j] #torch.Size([8, 800, 15625])
    C = x.shape[1]
    cols = cols.permute(1, 2, 0).contiguous().view(field_height * field_width * C, -1)
    return cols