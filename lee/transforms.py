import torch
import torch.nn.functional as F
import numpy as np

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        ix_nw = IW - 1 - (IW - 1 - ix_nw.abs()).abs()
        iy_nw = IH - 1 - (IH - 1 - iy_nw.abs()).abs()

        ix_ne = IW - 1 - (IW - 1 - ix_ne.abs()).abs()
        iy_ne = IH - 1 - (IH - 1 - iy_ne.abs()).abs()

        ix_sw = IW - 1 - (IW - 1 - ix_sw.abs()).abs()
        iy_sw = IH - 1 - (IH - 1 - iy_sw.abs()).abs()

        ix_se = IW - 1 - (IW - 1 - ix_se.abs()).abs()
        iy_se = IH - 1 - (IH - 1 - iy_se.abs()).abs()

    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.reshape(N, C, H, W) * nw.reshape(N, 1, H, W)
        + ne_val.reshape(N, C, H, W) * ne.reshape(N, 1, H, W)
        + sw_val.reshape(N, C, H, W) * sw.reshape(N, 1, H, W)
        + se_val.reshape(N, C, H, W) * se.reshape(N, 1, H, W)
    )

    return out_val


def img_like(img_shape):
    if len(img_shape) == 4 and img_shape[2] != 1 and img_shape[3] != 1:
        return True

    elif len(img_shape) == 3 and img_shape[1] != 1:
        return True
        
    else:
        return False


def count_tokens(img_shape):
    # no cls token　
    if len(img_shape) == 4 and img_shape[-2:] != (1, 1): return 0

    # cls token
    elif len(img_shape) == 3:
        # no cls token
        if np.sqrt(img_shape[1]).is_integer():  return 0
        # one cls token
        elif np.sqrt(img_shape[1] - 1).is_integer():  return 1 
        # two cls tokens
        elif np.sqrt(img_shape[1] - 2).is_integer(): return 2

    # error
    else: raise ValueError(f'Unsupported img_shape argument received: {img_shape}.')


def bnc2bchw(bnc, hw, num_tokens):
    # seperate an image part from cls tokens
    tokens = bnc[:, :num_tokens, :]
    img = bnc[:, num_tokens:, :]

    # reshape config
    b, n, c = bnc.shape
    if hw is not None:
        try:
            h, w = hw
        except:
            raise ValueError(f'hw argument must be a tupple of two integers')
    else:
        h = w = int(np.sqrt(n-num_tokens))
    
    return img.reshape(b, h, w, c).permute(0, 3, 1, 2), tokens


def bchw2bnc(bchw, tokens):
    b, c, h, w = bchw.shape
    n = h * w
    bnc = bchw.permute(0, 2, 3, 1).reshape(b, n, c)
    return torch.cat([tokens, bnc], dim=1)  # assumes tokens are at the start


def affine_transform(affineMatrices, img, **kwargs):
    assert img_like(img.shape), f'img argument must be imagelike, \n\
    1. len(img_shape) == 4 and img_shape[-2:] != (1, 1), \n\
    2. len(img_shape) == 3 and img_shape[1] != 1. \n\
    Actual: {img.shape}'

    # unpack kwargs
    hw = kwargs['hw'] if 'hw' in kwargs.keys() else None
    num_tokens = kwargs['num_tokens'] if 'num_tokens' in kwargs.keys() else None
    
    if len(img.shape) == 3:
        if num_tokens is None: 
            num_tokens = count_tokens(img.shape)

        assert (hw is not None) or (num_tokens is not None), f'Can not reshape data for transformation.'
        x, tokens = bnc2bchw(img, hw, num_tokens)
        
    else:
        x = img
        
    flowgrid = F.affine_grid(
        affineMatrices, size=x.size(), align_corners=True
    )  # .double()
    # uses manual grid sample implementation to be able to compute 2nd derivatives
    # img_out = F.grid_sample(img, flowgrid,padding_mode="reflection",align_corners=True)
    transformed = grid_sample(x, flowgrid)
    
    if len(img.shape) == 3:
        transformed = bchw2bnc(transformed, tokens)
        
    return transformed


def translate(img, t, axis="x", **kwargs):
    """Translates an image by a fraction of the size (sx,sy) in (0,1)"""
    batch_size = img.shape[0]
    device = img.device

    # create Matrices for translation
    affineMatrices = torch.zeros(batch_size, 2, 3).to(device)
    affineMatrices[:, 0, 0] = 1
    affineMatrices[:, 1, 1] = 1
    if axis == "x":
        affineMatrices[:, 0, 2] = t
    else:
        affineMatrices[:, 1, 2] = t
        
    return affine_transform(affineMatrices, img, **kwargs)


def rotate(img, angle, **kwargs):
    """Rotates an image by angle"""
    batch_size = img.shape[0]
    device = img.device
    
    affineMatrices = torch.zeros(batch_size, 2, 3).to(device)
    affineMatrices[:, 0, 0] = torch.cos(angle)
    affineMatrices[:, 0, 1] = torch.sin(angle)
    affineMatrices[:, 1, 0] = -torch.sin(angle)
    affineMatrices[:, 1, 1] = torch.cos(angle)
    return affine_transform(affineMatrices, img, **kwargs)


def shear(img, t, axis="x", **kwargs):
    """Shear an image by an amount t"""
    batch_size = img.shape[0]
    device = img.device
    
    affineMatrices = torch.zeros(batch_size, 2, 3).to(device)
    affineMatrices[:, 0, 0] = 1
    affineMatrices[:, 1, 1] = 1
    if axis == "x":
        affineMatrices[:, 0, 1] = t
        affineMatrices[:, 1, 0] = 0
    else:
        affineMatrices[:, 0, 1] = 0
        affineMatrices[:, 1, 0] = t
    return affine_transform(affineMatrices, img, **kwargs)


def stretch(img, x, axis="x", **kwargs):
    """Stretch an image by an amount t"""
    batch_size = img.shape[0]
    device = img.device
    
    affineMatrices = torch.zeros(batch_size, 2, 3).to(device)
    if axis == "x":
        affineMatrices[:, 0, 0] = 1 * (1 + x)
    else:
        affineMatrices[:, 1, 1] = 1 * (1 + x)
    return affine_transform(affineMatrices, img, **kwargs)


def hyperbolic_rotate(img, angle, **kwargs):
    batch_size = img.shape[0]
    device = img.device
    
    affineMatrices = torch.zeros(batch_size, 2, 3).to(device)
    affineMatrices[:, 0, 0] = torch.cosh(angle)
    affineMatrices[:, 0, 1] = torch.sinh(angle)
    affineMatrices[:, 1, 0] = torch.sinh(angle)
    affineMatrices[:, 1, 1] = torch.cosh(angle)
    return affine_transform(affineMatrices, img, **kwargs)


def scale(img, s, **kwargs):
    batch_size = img.shape[0]
    device = img.device
    
    affineMatrices = torch.zeros(batch_size, 2, 3).to(device)
    affineMatrices[:, 0, 0] = 1 - s
    affineMatrices[:, 1, 1] = 1 - s
    return affine_transform(affineMatrices, img, **kwargs)


def saturate(img, t):
    img = img.clone()
    img *= 1 + t
    return img