import torch
import torch.nn.functional as F
import pandas as pd
from .lie_derivs import *



def get_equivariance_metrics(minibatch, model, model_type='classifier', output_type=torch.Tensor, output_key=None):
    device = next(model.parameters()).device
    x, _ = minibatch
    
    if device.type == 'cuda':
        x = x.to(device)

    # check model_type and output
    ## the model is a whole network
    if model_type == 'classifier':
        if output_type is torch.Tensor:
            m = lambda x: model(x)
            model_probs = lambda x: F.softmax(m(x), dim=-1)           
        elif output_type is dict:
            m = lambda x: model(x)[output_key]
            model_probs = lambda x: F.softmax(m(x), dim=-1)
        else:
            m = lambda x: model(x)[0]
            model_probs = lambda x: F.softmax(m(x), dim=-1)
        
    elif model_type == 'encoder':
        assert output_key is not None, 'Output_key for encoder output is required except for None.'
        m = lambda x: model(x)[output_key].reshape(x.shape[0], -1)
        model_probs = lambda x: m(x)

    ## the model is a subnetwork
    elif model_type == 'subnetwork':
        if output_type is torch.Tensor:
            model_probs = lambda x: model(x)       
        elif output_type is dict:
            model_probs = lambda x: model(x)[output_key]
        else:
            model_probs = lambda x: model(x)[0]

    else:
        msg = f"Unexpected model_type argument was received. expected: \'classifier', 'encoder' or 'subnetwork'\. actural: {model_type}"
        raise ValueError(msg)

    # calculate a model output
    model_out = model_probs(x)

    derivs = {
        "trans_x_deriv": translation_lie_deriv(model_probs, x, model_out, axis="x"),
        "trans_y_deriv": translation_lie_deriv(model_probs, x, model_out, axis="y"),
        "rot_deriv": rotation_lie_deriv(model_probs, x, model_out),
        "shear_x_deriv": shear_lie_deriv(model_probs, x, model_out, axis="x"),
        "shear_y_deriv": shear_lie_deriv(model_probs, x, model_out, axis="y"),
        "stretch_x_deriv": stretch_lie_deriv(model_probs, x, model_out, axis="x"),
        "stretch_y_deriv": stretch_lie_deriv(model_probs, x, model_out, axis="y"),

        "saturate_err": saturate_lie_deriv(model_probs, x, model_out),
    }

    def compute_LEE(derivative):
        assert len(derivative.shape) > 1, \
        f'the input was not a result from batch process. The fist axis are expected batch indeces.'

        ## get the batch size
        B = derivative.shape[0]
        
        ## compute expectations and scale by the output size.
        ## these computations are compatible with computation of mean after abs.
        return derivative.reshape(B, -1).abs().cpu().data.numpy().mean(-1)
    
    metrics = {key: pd.Series(compute_LEE(d)) for key, d in derivs.items()}
    df = pd.DataFrame.from_dict(metrics)

    return df
