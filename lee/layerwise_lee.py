import os
import tqdm
import argparse
import functools
from functools import partial

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lie_derivs import *
from .transforms import *


HOOK_CONTEXT = None # global variable for hooks

class _CalledHookContext:
    forward = False
    count = 0

    def __init__(self):
        pass

    def __enter__(self):
        self.forward = True

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.forward = False
        
# wrapper for initialization of a global variable
def _in_context(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """
        Decorator for use global variables such as HOOK_CONTEXT.
        Initialize HOOK_CONTEXT before execute functions
        """
        global HOOK_CONTEXT
        HOOK_CONTEXT = _CalledHookContext()
        return f(*args, **kwargs)

    return wrapper
        

class store_inputs:

    def __init__(self, batch_size, lie_deriv, use_forward_options):
        self.batch_size = batch_size
        self.lie_deriv = lie_deriv
        self.use_forward_options = use_forward_options
        self._in_main_forward = True # condition for updating variables and calculating lee
        

    def __call__(self, module, inputs, outputs):

        if self._in_main_forward is True:
        
            if not hasattr(module, "_lie_norm_sum"):
                self.set_variables(module)

            # calculate lee
            d_lee = self.calc_lee(module, inputs, outputs)
            module._lie_deriv_output = d_lee
            
            # flag on for next main forward
            self._in_main_forward = not self._in_main_forward

            # count the number of modules
            module._module_number = HOOK_CONTEXT.count
            HOOK_CONTEXT.count += 1

    # set variables at the module for results of lie-deriv
    def set_variables(self, module):
        
        module._module_number = 0
        module._lie_norm_sum = 0.0
        module._lie_norm_sum_sq = 0.0
        module._lie_deriv_output = 0.0
        module._num_probes = 0
        module._estimation = True # avoid double estimation at skip connections

    # calculate lie_deriv
    def calc_lee(self, module, inputs, outputs):
        
        # lie deriv calculation also call hook
        # but updating variables is undesirable
        self._in_main_forward = not self._in_main_forward

        # clone the input tensor
        x = inputs[0].clone().detach()
        x = x + torch.zeros_like(x)

        # output tensors must have the same size with the bath size
        # In some case, outputs = (output tensor, None),
        # e.g., 'google/vit-base-patch16-224'
        assert len(outputs) == self.batch_size \
            or len(outputs[0]) == self.batch_size, \
            "Outputs has an unexpected size"
        # clone the output tensor
        if len(outputs) == self.batch_size:
            y = outputs.clone().detach()
        else:
            y = outputs[0].clone().detach()

        if (self.use_forward_options is True) and len(inputs) > 1:
            p = inputs[1:]
            return self.lie_deriv(module, x, y, *p)
            
        else:
            return self.lie_deriv(module, x, y)


def store_estimator(module, grad_input, grad_output):

    # if calling is during backward and the first time
    if (HOOK_CONTEXT.forward is False) and (module._estimation is True):
        grad = grad_output[0].clone().detach()
    
        # estimate a piece of equivariance error 
        def estimator(g, lxf):
            len_x = g.shape[0]
            e = (g*lxf).reshape(len_x, -1)
            e = e.sum(-1)
            e = e**2
            return e.cpu().data.numpy()
        estimator = estimator(grad, module._lie_deriv_output)
        
        module._lie_norm_sum += estimator
        module._lie_norm_sum_sq += estimator**2
        module._num_probes += 1

        module._estimation = False


from timm.models.vision_transformer import Attention as A1
# from timm.models.vision_transformer_wconvs import Attention as A2
from timm.models.mlp_mixer import MixerBlock, Affine, SpatialGatingBlock
from timm.models.layers import PatchEmbed, Mlp, DropPath, BlurPool2d

# from timm.models.layers import FastAdaptiveAvgPool2d,AdaptiveAvgMaxPool2d
# ad hoc: EvoNormBatch2d is removed from timm at a commit:78912b6
# from timm.models.layers import GatherExcite, EvoNormBatch2d
from timm.models.layers import GatherExcite
from timm.models.senet import SEModule
from timm.models._efficientnet_blocks import SqueezeExcite
from timm.models.convit import MHSA, GPSA

from transformers.models.vit.modeling_vit import ViTSdpaAttention

#from torchlibrosa.stft import LogmelFilterBank
from torchlibrosa.augmentation import DropStripes

leaflist = (
    A1,
    # A2,
    MixerBlock,
    Affine,
    SpatialGatingBlock,
    PatchEmbed,
    Mlp,
    DropPath,
    BlurPool2d,
    ViTSdpaAttention,
)
# leaflist += (nn.AdaptiveAvgPool2d,nn.MaxPool2d,nn.AvgPool2d)
leaflist += (
    GatherExcite,
    # EvoNormBatch2d,
    nn.BatchNorm2d,
    nn.BatchNorm1d,
    nn.LayerNorm,
    nn.GroupNorm,
    SEModule,
    SqueezeExcite,
)
leaflist += (MHSA, GPSA)

def is_leaf(m):
    return (not hasattr(m, "children") or not list(m.children())) or isinstance(
        m, leaflist
    )


def is_excluded(m):
    excluded_list = (nn.Dropout,
                     nn.Identity,
                     nn.Flatten,
                     nn.Sequential,
                     #LogmelFilterBank,
                     DropStripes
                    )
    return isinstance(m, excluded_list)


def selective_apply(m, fn, output_collection=None):
    if output_collection is None:
        output_collection = []

    if is_leaf(m):
        if not is_excluded(m):
            output_collection.append(fn(m))
    else:
        for c in m.children():
            selective_apply(c, fn, output_collection)

    return output_collection

    
def apply_hooks(model, batch_size, lie_deriv_type, use_forward_options=False, **lie_deriv_config):
    lie_deriv = {
        "translation": translation_lie_deriv,
        "rotation": rotation_lie_deriv,
        "hyper_rotation": hyperbolic_rotation_lie_deriv,
        "scale": scale_lie_deriv,
        "saturate": saturate_lie_deriv,
    }[lie_deriv_type]

    if len(lie_deriv_config) != 0:
        lie_deriv = partial(lie_deriv, **lie_deriv_config)

    # register hooks
    handles = []
    ## forward hook
    def register_hook_obj(module, cls, **kwargs):
        hook = cls(**kwargs)
        module.register_forward_hook(hook)
    func = partial(register_hook_obj,
                   cls=store_inputs,
                   batch_size=batch_size,
                   lie_deriv=lie_deriv,
                   use_forward_options=use_forward_options
                  )
    h = selective_apply(model, func)
    handles.extend(h)

    ## backward hook
    func = lambda m: m.register_backward_hook(store_estimator)
    h = selective_apply(model, func)
    handles.extend(h)

    return handles


def initialize_variables(module):
    module._module_number = 0
    module._lie_norm_sum = 0.0
    module._lie_norm_sum_sq = 0.0
    module._num_probes = 0
    module._in_main_forward = True
    module._estimation = True


def unset_variables(module):
    try:
        del module._lie_norm_sum
        del module._lie_norm_sum_sq
        del module._num_probes
        del module._lie_deriv_output
        del module._estimation
        
    except AttributeError:
        pass


def normalized_randn_like(tensor, ax=0):
    # get the dimension of the vector
    if ax in range(tensor.dim()):
        d = tensor.numel()//tensor.shape[0]
    else:
        d = tensor.numel()

    # return a vector whose entries are sampled independently from N(0, 1/sqrt(d))
    return torch.randn_like(tensor)/torch.sqrt(torch.tensor(d, dtype=tensor.dtype))

    
def compute_output_variation_with_perturbation(model, x, perturbation=normalized_randn_like):

    dx = perturbation(x)
    x_noise = x + dx        
    
    with torch.no_grad():
        y = model(x)
        y_noise = model(x_noise)

    def norm(x):
        x = x.reshape(x.shape[0], -1)
        return torch.linalg.vector_norm(x, dim=1)

    dy = norm(y-y_noise)
    dx = norm(dx)
    #dx = np.ones(x.shape[0])
    
    if isinstance(dy, torch.Tensor):
        dy = dy.to('cpu').detach().numpy()
    if isinstance(dx, torch.Tensor):
        dx = dx.to('cpu').detach().numpy()

    assert dy.shape[0] == dx.shape[0], 'Passed norm fucn must compute norms of each output.'

    return dy, dx


@_in_context
def compute_equivariance_attribution(model, img_batch, num_probes=100, top_layer='classifier', output_type=torch.Tensor, output_key=None, normalize=False):
    
    # check top_layer and output
    if top_layer == 'classifier':
        if output_type is torch.Tensor:
            m = lambda x: model(x)
            model_fn = lambda x: F.softmax(m(x), dim=-1)           
        elif output_type is dict:
            m = lambda x: model(x)[output_key]
            model_fn = lambda x: F.softmax(m(x), dim=-1)
        else:
            m = lambda x: model(x)[0]
            model_fn = lambda x: F.softmax(m(x), dim=-1)
        
    elif top_layer == 'encoder':
        assert output_key is not None, 'Output_key for encoder output is required except for None.'
        m = lambda x: model(x)[output_key].reshape(x.shape[0], -1)
        model_fn = lambda x: F.softmax(m(x), dim=-1)

    else:
        msg = f"Unexpected top_layer argument was received. expected: \'classifier', 'encoder'\. actural: {top_layer}"
        raise ValueError(msg)
        

    # main 
    all_errs = []
    order = []
    for j in range(num_probes):

        # calcualte lie deriv by registered hooks
        with HOOK_CONTEXT:
            y = model_fn(img_batch)
        # perturbation
        z = torch.randn(y.shape[0], y.shape[1]).to(y.device)
        loss = (z*y).sum()
        loss.backward()

        # collect errors in dictionary
        errs = {}
        for name, module in model.named_modules():
            
            if hasattr(module, "_lie_norm_sum"):
                assert module._estimation is False, \
                    f'Unestimated layer was found: {module._module_number} {name}. Its backward hook may not be called.'
                lie_norm = module._lie_norm_sum/module._num_probes 
                key = (name, module.__class__.__name__, module._module_number)
                
                errs[key] = lie_norm
                
            else:
                pass

        # add each errors to a container
        all_errs.append(errs)

        # to next iter
        HOOK_CONTEXT.count = 0
        model.zero_grad()
        selective_apply(model, initialize_variables)

    
    # initialize a dictionary for aggregation
    aggregated_data = {key: [] for key in all_errs[0].keys()}

    
    # aggregate data
    for errs in all_errs:
        for key in errs.keys():
            aggregated_data[key].extend(errs[key])

    
    # to pandas.DataFrame
    df = pd.DataFrame(aggregated_data)

    
    # compute differentials for normalization
    if normalize:
        #dy, dx = compute_output_variation_with_perturbation(model_fn, img_batch)
        dy, dx = compute_output_variation_with_perturbation(m, img_batch)
    else:
        batch_size = img_batch.shape[0]
        dy, dx = np.ones(batch_size), np.ones(batch_size)
    print(f'|y - y_noise| = {dy}')
    print(f'|dx| = {dx}')
        
    
    # unset variables
    model.apply(unset_variables)

    
    return df, dy, dx


    