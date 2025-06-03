import copy
import torch
from .transforms import *
from functools import partial

def jvp(f, x, u):
    """Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).
    Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u"""
    with torch.enable_grad():
        y = f(x)

        # Dummy variable (could take any value)
        v = torch.ones_like(y, requires_grad=True)
        vJ = torch.autograd.grad(y, [x], [v], create_graph=True)
        Ju = torch.autograd.grad(vJ, [v], [u], create_graph=True)
        
        return Ju[0]


def translation_lie_deriv(model, inp_imgs, out_imgs, inp_options=None, axis="x", **kwargs):
    """Lie derivative of model with respect to translation vector, output can be a scalar or an image"""
    
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)


    def shifted_model(t):
        # print("Input shape",inp_imgs.shape)
        shifted_img = translate(inp_imgs, t, axis, **kwargs)

        if inp_options is not None:
            z = model(shifted_img, inp_options)
        else:
            z = model(shifted_img)
        
        # print("Output shape",z.shape)
        # if model produces an output image, shift it back
        if hasattr(z, 'shape') and img_like(z.shape):
            z = translate(z, -t, axis, **kwargs)
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = translate(z[0], -t, axis, **kwargs)
            
        # print('zshape',z.shape)
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(shifted_model, t, torch.ones_like(t, requires_grad=True))
    # print('Liederiv shape',lie_deriv.shape)
    # print(model.__class__.__name__)
    # print('')
    
    return lie_deriv


def rotation_lie_deriv(model, inp_imgs, out_imgs, inp_options=None, **kwargs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)

    def rotated_model(t):
        rotated_img = rotate(inp_imgs, t, **kwargs)

        if inp_options is not None:
            z = model(rotated_img, inp_options)
        else:
            z = model(rotated_img)
            
        if hasattr(z, 'shape') and img_like(z.shape):
            z = rotate(z, -t, **kwargs)
            
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = rotate(z, -t, **kwargs)
        
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def hyperbolic_rotation_lie_deriv(model, inp_imgs, out_imgs, inp_options=None, **kwargs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)

    def rotated_model(t):
        rotated_img = hyperbolic_rotate(inp_imgs, t, **kwargs)

        if inp_options is not None:
            z = model(rotated_img, inp_options)
        else:
            z = model(rotated_img)

        if hasattr(z, 'shape') and img_like(z.shape):
            z = hyperbolic_rotate(z, -t, **kwargs)
            
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = hyperbolic_rotate(z, -t, **kwargs)
        
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
    return lie_deriv


def scale_lie_deriv(model, inp_imgs, out_imgs, inp_options=None, **kwargs):
    """Lie derivative of model with respect to rotation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)

    def scaled_model(t):
        scaled_img = scale(inp_imgs, t, **kwargs)

        if inp_options is not None:
            z = model(scaled_img, inp_options)
        else:
            z = model(scaled_img)

        if img_like(z.shape):
            z = scale(z, -t, **kwargs)

        if hasattr(z, 'shape') and img_like(z.shape):
            z = scale(z, -t, **kwargs)
            
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = scale(z, -t, **kwargs)
        
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(scaled_model, t, torch.ones_like(t))
    return lie_deriv


def shear_lie_deriv(model, inp_imgs, out_imgs, inp_options=None, axis="x", **kwargs):
    """Lie derivative of model with respect to shear, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)

    def sheared_model(t):
        sheared_img = shear(inp_imgs, t, axis, **kwargs)

        if inp_options is not None:
            z = model(sheared_img, inp_options)
        else:
            z = model(sheared_img)


        if hasattr(z, 'shape') and img_like(z.shape):
            z = shear(z, -t, axis, **kwargs)
            
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = shear(z, -t, axis, **kwargs)
        
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(sheared_model, t, torch.ones_like(t))
    return lie_deriv


def stretch_lie_deriv(model, inp_imgs, out_imgs, axis="x", **kwargs):
    """Lie derivative of model with respect to stretch, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)

    def stretched_model(t):
        stretched_img = stretch(inp_imgs, t, axis, **kwargs)
        z = model(stretched_img)

        if hasattr(z, 'shape') and img_like(z.shape):
            z = stretch(z, -t, axis, **kwargs)
            
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = stretch(z, -t, axis, **kwargs)
        
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(stretched_model, t, torch.ones_like(t))
    return lie_deriv


def saturate_lie_deriv(model, inp_imgs, out_imgs, **kwargs):
    """Lie derivative of model with respect to saturation, assumes scalar output"""
    if not img_like(inp_imgs.shape):
        return torch.zeros_like(out_imgs)

    def saturated_model(t):
        saturated_img = saturate(inp_imgs, t, **kwargs)
        z = model(saturated_img)

        if hasattr(z, 'shape') and img_like(z.shape):
            z = saturate(z, -t, **kwargs)
            
        if not hasattr(z, 'shape') and img_like(z[0].shape):
            z = saturate(z, -t, **kwargs)
            
        return z

    t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
    lie_deriv = jvp(saturated_model, t, torch.ones_like(t))
    return lie_deriv

