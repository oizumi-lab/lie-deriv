from email import message_from_bytes
from mailbox import Message
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from functools import partial

import sys
sys.path.append("stylegan3")
from stylegan3.metrics.equivariance import (
    apply_integer_translation,
    apply_fractional_translation,
    apply_fractional_rotation,
)

from .transforms import (
    translate,
    rotate,
    img_like,
)

from .layerwise_lee import compute_output_variation_with_perturbation as perturbation_error


def _calc_mse_with_chunks(x1, x2, num_chunks):
    # torch.no_grad() save GPU memory use
    with torch.no_grad():
        mse_list = []
        x1_chunks = torch.chunk(x1, num_chunks, dim=0)
        x2_chunks = torch.chunk(x2, num_chunks, dim=0)
        
        for dx1, dx2 in zip(x1_chunks, x2_chunks):
            dims = list(range(1, dx1.dim()))
            mse_chunk = ((dx1 - dx2)**2).mean(dim=dims)
            mse_list.append(mse_chunk)

    return torch.cat(mse_list)


def EQ_T(model_probs, inp_imgs, model_out, mode='integer', translate_max=0.125, num_probes=10, num_chunks=10):
    N, C, H, W = inp_imgs.shape
    device = inp_imgs.device

    ## Check the mode argument and define a translation
    assert mode in ['integer', 'fractional'], "mode must be \'integer\' or \'fractional\'"
    if mode == 'integer':
        apply_translation = apply_integer_translation
    else:
        apply_translation = apply_fractional_translation

    ## Generate shifted images
    ### random translation amounts for each sample (N, num_probes, 2)
    ### translation amounts are sumpled from [-translate_max, translate_max]^2
    t = (torch.rand(N, num_probes, 2, device=device) * 2 - 1) * translate_max
    if mode == 'integer':
        t = (t * H * W).round() / (H * W)
    t = t.view(N, num_probes, 2)

    ### Duplicate images for probes
    x = inp_imgs.unsqueeze(1).repeat(1, num_probes, 1, 1, 1).view(N*num_probes, C, H, W)

    ### Prepare for batched translations
    tx = t[:, :, 0].view(-1)  # Flatten (N * num_probes,)
    ty = t[:, :, 1].view(-1)

    # Apply translations in a loop
    translated_imgs = []
    for dx, dtx, dty in zip(x, tx, ty):
        dx = dx.unsqueeze(0)
        dx, _ = apply_translation(dx, dtx, dty)
        translated_imgs.append(dx)

    # Combine all translated images back into a batch
    x = torch.cat(translated_imgs, dim=0)  # Shape: (N * num_probes, C, H, W)

    ## feed shifted image to the model: (N * P, C', H', W')
    y_translated = model_probs(x)

    ## Calculate MSE between original and shifted model outputs
    ### Dupulicate the original model output for probes
    y = model_out.repeat_interleave(num_probes, dim=0)  # (N * P, C', H', W')

    ### To measure equivariance, apply translation to y if it has an image like shape
    assert hasattr(y, 'shape')
    if img_like(y.shape):
        if y.dim() == 3:
            y = y.unsqueeze(1) #(N * P, 1, H', W')
            unsqueezed_flag = True
        else:
            unsqueezed_flag = False

        translated_y = []
        for dy, dtx, dty in zip(y, tx, ty):
            #unsqueeze so that dy.dim() == 4
            dy = dy.unsqueeze(0) #(1, 1, H', W')
            dy, _ = apply_translation(dy, dtx, dty)
            if unsqueezed_flag:
                dy = dy.squeeze(0) #(1, H', W')
            translated_y.append(dy)

        # update y to that applied translation
        translated_y = torch.cat(translated_y, dim=0)  # Shape: (N * num_probes, C, H, W)
    else:
        translated_y = y
        
    ### Calculate MSE: (N * P,)
    mse = _calc_mse_with_chunks(y_translated, translated_y, num_chunks)
    ### mean over probes
    mse = mse.view(N, num_probes).mean(dim=1)  # (N,)

    return mse  # shape [N]


def EQ_R(model_probs, inp_imgs, model_out, rotate_max=1.0, num_probes=10, num_chunks=10):
    N, C, H, W = inp_imgs.shape
    device = inp_imgs.device

    ## Generate rotated images
    ### Generate random rotation amounts for each sample (N, num_probes, 1)
    angles = (torch.rand(N, num_probes, 1, device=device)*2 - 1) * (rotate_max*np.pi)
    angles = angles.view(N*num_probes, 1)

    ### Duplicate images for probes
    x = inp_imgs.unsqueeze(1).repeat(1, num_probes, 1, 1, 1).view(N*num_probes, C, H, W)

    ### Apply fractional rotation in batch
    rotated_imgs = []
    for dx, da in zip(x, angles):
        dx = dx.unsqueeze(0)
        dx, _ = apply_fractional_rotation(dx, da)
        rotated_imgs.append(dx)
    
    # Combine all translated images back into a batch
    x = torch.cat(rotated_imgs, dim=0)  # Shape: (N * num_probes, C, H, W)

    ## feed shifted image to the model: (N * P, C', H', W')
    y_rotated = model_probs(x)

    ## Calculate MSE between original and shifted model outputs
    ### Dupulicate the original model output for probes
    y = model_out.repeat_interleave(num_probes, dim=0)  # (N * P, C', H', W')

    ### To measure equivariance, apply rotation to y if it has an image like shape
    assert hasattr(y, 'shape')
    if img_like(y.shape):
        if y.dim() == 3:
            y = y.unsqueeze(1) #(N * P, 1, H', W')
            unsqueezed_flag = True
        else:
            unsqueezed_flag = False
            
        rotated_y = []
        for dy, da in zip(y, angles):
            #unsqueeze so that dy.dim() == 4
            dy = dy.unsqueeze(0) #(1, 1, H', W')
            dy, _ = apply_fractional_rotation(dy, da)
            if unsqueezed_flag:
                dy = dy.squeeze(0) #(1, H', W')
            rotated_y.append(dy)

        # update y to that applied translation
        rotated_y = torch.cat(rotated_y, dim=0)  # Shape: (N * num_probes, C, H, W)
    else:
        rotated_y = y

    ### Calculate MSE: (N * P,)
    mse = _calc_mse_with_chunks(y_rotated, rotated_y, num_chunks)
    ### mean over probes
    mse = mse.view(N, num_probes).mean(dim=1)  # (N,)

    return mse  # shape [N]


def translation_sample_equivariance(model, inp_imgs, model_out, axis='x', eta=0.01, num_probes=10, num_chunks=10):
    N, C, H, W = inp_imgs.shape
    device = inp_imgs.device

    ## Generate shifted images
    ### random translation amounts for each sample (N, num_probes, )
    ### translation amounts are sumpled from [-eta, eta]
    t = (2*eta)*torch.rand(N*num_probes, device=device) - eta

    ### Duplicate images for probes
    x = inp_imgs.unsqueeze(1).repeat(1, num_probes, 1, 1, 1).view(N*num_probes, C, H, W)

    ### get translated model outputs
    translate_ax = partial(translate, axis=axis)
    x_translated = translate_ax(x, t)
    y_translated = model(x_translated)

    ## Calculate MSE between original and shifted model outputs
    ### Dupulicate the original model output for probes
    y = model_out.repeat_interleave(num_probes, dim=0)  # (N * P, C', H', W')

    ### To measure equivariance, 
    ### apply translation to y if it has an image like shape
    assert hasattr(y, 'shape')
    if img_like(y.shape):
        translated_y = translate_ax(y, -t)
    else:
        translated_y = y

    ### Calculate MSE: (N * P,)
    mse = _calc_mse_with_chunks(y_translated, translated_y, num_chunks)
    ### mean over probes
    mse = mse.view(N, num_probes).mean(dim=1)  # (N,)

    return mse.unsqueeze(1)  # shape [N, 1]


def rotation_sample_equivariance(model, inp_imgs, model_out, eta=np.pi/360, num_probes=10, num_chunks=10):
    N, C, H, W = inp_imgs.shape
    device = inp_imgs.device

    ## Generate rotated images
    ### random rotaion amounts for each sample (N, num_probes, )
    ### roatation amounts are sumpled from [-eta, eta]
    t = (2*eta)*torch.rand(N*num_probes, device=device) - eta

    ### Duplicate images for probes
    x = inp_imgs.unsqueeze(1).repeat(1, num_probes, 1, 1, 1).view(N*num_probes, C, H, W)

    ### get rotated model outputs
    x_rotated = rotate(x, t)
    y_rotated = model(x_rotated)

    ## Calculate MSE between original and shifted model outputs
    ### Dupulicate the original model output for probes
    y = model_out.repeat_interleave(num_probes, dim=0)  # (N * P, C', H', W')

    ### To measure equivariance, apply translation to y if it has an image like shape
    assert hasattr(y, 'shape')
    if img_like(y.shape):
        rotated_y = rotate(y, t)
    else:
        rotated_y = y

    ### Calculate MSE: (N * P,)
    mse = _calc_mse_with_chunks(y_rotated, y, num_chunks)
    ### mean over probes
    mse = mse.view(N, num_probes).mean(dim=1)  # (N,)

    return mse.unsqueeze(1)  # shape [N, 1]


def get_equivariance_metrics(minibatch, model, num_probes=200, model_type='classifier', output_type=torch.Tensor, output_key=None):
    device = next(model.parameters()).device
    x, y = minibatch
    N = x.shape[0] #batch size

    if device.type == 'cuda':
        x, y = x.to(device), y.to(device)
        equal = lambda x1, x2: torch.eq(x1, x2).cpu().data.numpy()
    else:
        equal = np.equal

    # Get model outputs
    if model_type == 'classifier':
        ## Add a softmax on top for calculating accuracy
        if output_type is torch.Tensor:
            m = lambda img: model(img)
            model_probs = lambda img: F.softmax(m(img), dim=-1)           
        elif output_type is dict:
            m = lambda img: model(img)[output_key]
            model_probs = lambda img: F.softmax(m(img), dim=-1)
        else:
            m = lambda img: model(img)[0]
            model_probs = lambda img: F.softmax(m(img), dim=-1)

        ## Get model output
        model_out = model_probs(x)
        # Ensure model_out has at least 2 dimensions, batch size x num_classes
        if model_out.dim() < 2:
            raise ValueError(f"Expected model output to have at least 2 dimensions, but got {model_out.dim()} dimensions.")

        pred = lambda out: out.argmax(dim=1)
        yhat = pred(model_out)

        ## Calculate accuracy
        acc = (yhat == y).cpu().float().data.numpy()

        ## caluculate perturbation error
        dy, _ = perturbation_error(model_probs, x)

    elif model_type == 'encoder':
        assert output_key is not None, 'Output_key for encoder output is required except for None.'
        m = lambda img: model(img)[output_key].reshape(img.shape[0], -1)
        model_probs = lambda img: m(img)

        ## Get model output
        model_out = model_probs(x)
        pred = lambda out: out  # Identity because encoder output is not a probability
        yhat = pred(model_out)

        ## If model_type is encoder, accuracy can not be calculated
        acc = np.full(N, np.nan)

        ## caluculate perturbation error
        dy, _ = perturbation_error(model_probs, x)

        ## Equal function for encoding consistency
        equal = lambda x1, x2: [torch.equal(dx1, dx2) 
                                 for dx1, dx2 in zip(x1, x2)]

    elif model_type == 'subnetwork':
        if output_type is torch.Tensor:
            model_probs = lambda x: model(x)       
        elif output_type is dict:
            model_probs = lambda x: model(x)[output_key]
        else:
            model_probs = lambda x: model(x)[0]
 
        ## Get model output
        model_out = model_probs(x)
        pred = lambda out: out  # Identity because a subnetwork output is not a probability
        yhat = pred(model_out)

        ## If model_type is subnetwork, accuracy is undifined.
        acc = np.full(N, np.nan)

        ## caluculate perturbation error
        dy, _ = perturbation_error(model_probs, x)

        ## Equal function for output consistency
        equal = lambda x1, x2: [torch.equal(dx1, dx2) 
                                 for dx1, dx2 in zip(x1, x2)]

    else:
        msg = f"Unexpected model_type argument was received. expected: \'classifier', 'encoder' or 'subnetwork'\. actual: {model_type}"
        raise ValueError(msg)

    metrics = {}
    metrics["acc"] = pd.Series(acc)
    metrics["dy"] = pd.Series(dy)

    with torch.no_grad():

        ## Calculate prediction consistency
        for shift_x in range(8):
            rolled_img = torch.roll(x, shift_x, 2)
            rolled_out = model_probs(rolled_img)
            rolled_yhat = pred(rolled_out)
            consistency = equal(rolled_yhat, yhat)
            metrics["consistency_x" + str(shift_x)] = pd.Series(consistency)

        for shift_y in range(8):
            rolled_img = torch.roll(x, shift_y, 3)
            rolled_out = model_probs(rolled_img)
            rolled_yhat = pred(rolled_out)
            consistency = equal(rolled_yhat, yhat)
            metrics["consistency_y" + str(shift_y)] = pd.Series(consistency)

        ### Calculate equivariance proposed by Karras et al., 2021
        # Vectorized EQ metrics
        eq_t = EQ_T(
            model_probs, x, model_out, mode='integer', num_probes=num_probes
        ).cpu().data.numpy()  # shape [N]
        eq_t_frac = EQ_T(
            model_probs, x, model_out, mode='fractional', num_probes=num_probes
        ).cpu().data.numpy()  # shape [N]
        eq_r = EQ_R(
            model_probs, x, model_out, num_probes=num_probes
        ).cpu().data.numpy()  # shape [N]

        metrics["eq_t"] = pd.Series(eq_t)
        metrics["eq_t_frac"] = pd.Series(eq_t_frac)
        metrics["eq_r"] = pd.Series(eq_r)

        ## Calculate expected group sample equivariance
        # Vectorized invariance metrics
        trans_x_sample = translation_sample_equivariance(
            model_probs, x, model_out, axis='x'
        ).cpu().data.numpy()  # shape [N, 1]
        trans_y_sample = translation_sample_equivariance(
            model_probs, x, model_out, axis='y'
        ).cpu().data.numpy()  # shape [N, 1]
        rotate_sample = rotation_sample_equivariance(
            model_probs, x, model_out
        ).cpu().data.numpy()  # shape [N, 1]

        metrics['trans_x_sample'] = pd.Series(trans_x_sample.flatten())
        metrics['trans_y_sample'] = pd.Series(trans_y_sample.flatten())
        metrics['rotate_sample'] = pd.Series(rotate_sample.flatten())

    df = pd.DataFrame.from_dict(metrics)
    return df