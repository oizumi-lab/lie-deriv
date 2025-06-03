import os
import sys
import tqdm
import copy
import argparse
import warnings
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# Jacobian of efficient attention is not implemented
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import timm

import numpy as np
import pandas as pd

sys.path.append('./lie-deriv')
import lee.layerwise_lee as lee
import lee.layerwise_other as other_metrics
from lee.loader import get_loaders



def convert_inplace_relu_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_inplace_relu_to_relu(child)


def set_requires_grad_True(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def normalize_errors(df, normalizing_constant):
    ## the first column, index=0, represents id
    ## the last two ones, index=-2, -1, represent image ids and model names
    cols = df.columns[1:-2]
    ## the top three rows represent module names, those types and those ids
    rows = df.index >= 3
    
    # devide
    def _to_numeric_and_divide(series, divisor=normalizing_constant):
        return pd.to_numeric(series, errors='coerce')/divisor

    df.loc[rows, cols] = df.loc[rows, cols].apply(_to_numeric_and_divide)
        
    return df


def get_layerwise(calc_errors, loader, 
                  model_name, num_imgs, num_probes, device):
    
    errlist = []
    differentials = []
    perturbations = []
    for idx, (x, _) in tqdm.tqdm(
        enumerate(loader), total=len(loader)
    ):
        if idx >= num_imgs:
            break
        
        img = x.to(device)
        errors, dy, dx = calc_errors(img)
        
        batch_size = len(x)
        img_idx_start,  img_idx_stop = idx*batch_size, (idx+1)*batch_size
        errors["img_idx"] = np.tile(np.arange(img_idx_start, img_idx_stop), num_probes)
        errlist.append(errors)
        
        differentials.extend(dy)
        perturbations.extend(dx)
            
    df = pd.concat(errlist, axis=0)
    df["model"] = model_name

    # compute a normalizing constant
    differentials = np.asarray(differentials)
    perturbations = np.asarray(perturbations)
    #print(differentials)
    #print(perturbations)
    
    #normalizing_constant = differentials.mean()/perturbations.mean()
    normalizing_constant = differentials.mean()
    print(f'normalizing_constant = {normalizing_constant}')

    # normalize
    df = normalize_errors(df, normalizing_constant)
    
    return df



# main
def main(args):
    warnings.simplefilter('ignore')

    # unpuck args
    OUT_DIR = args.out_dir
    MODEL_NAME = args.model_name
    UNTRAINED = args.untrained
    CHKPT_LIST = args.chkpt_list
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_IMGS = args.num_imgs
    NUM_PROBES = args.num_probes
    TRANSFORM = args.transform
    DEVICE = torch.device(type='cuda', index=args.gpu_id)
    NORMALIZE = args.normalize
    
    print(MODEL_NAME)
    print(TRANSFORM)
    print(DEVICE)
    
    # define available models
    with open(CHKPT_LIST, 'r') as f:
        chkpt_list = json.load(f)

    ADDITIONAL_MODELS = {'simclr_resnet50_1x': None, 'mae_vit_large_patch16': None,  'clip': None, 
                         'vit_small_patch16_224_untrainedlike': None,
                        }
    
    # load a model
    if UNTRAINED is False:
        if MODEL_NAME in chkpt_list:
            ## get a model configuration
            backborn = chkpt_list[MODEL_NAME]["backbone"]
            weights = chkpt_list[MODEL_NAME]["weights"]

            ## instantiate a model & load weights
            model = timm.create_model(backborn, pretrained=False)
            model.load_state_dict(torch.load(weights))

            ## get a model type and output key
            ## - flag for a branch in the calculation of LEE
            TOP_LAYER = chkpt_list[MODEL_NAME]["type"]
            OUTPUT_KEY = chkpt_list[MODEL_NAME]["outputkey"]

        elif MODEL_NAME in ADDITIONAL_MODELS:
            if MODEL_NAME == 'simclr_resnet50_1x':

                import json, torchvision
                with open(CHKPT_LIST, 'r') as f:
                    chkpt_list = json.load(f)

                weights = chkpt_list[MODEL_NAME]["weights"]
                weights = torch.load(weights)
                model = torchvision.models.resnet50(weights=weights)

                TOP_LAYER = 'classifier'
                OUTPUT_KEY = None
                
            elif MODEL_NAME == 'mae_vit_large_patch16':
                from transformers import ViTMAEModel
                model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
                model.config.mask_ratio = 0.0 # no maskikng

                TOP_LAYER = 'encoder'
                OUTPUT_KEY = 'last_hidden_state'
            
            elif MODEL_NAME == 'clip':
                from transformers import CLIPVisionModel
                model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

                TOP_LAYER = 'encoder'
                OUTPUT_KEY = 'pooler_output'
        
        else:
            model = timm.create_model(MODEL_NAME, pretrained=True)

            TOP_LAYER = 'classifier'
            OUTPUT_KEY = None # outtputs are expcted to be an object of torch.Tensor

    else:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1000)

        TOP_LAYER = 'classifier'
        OUTPUT_KEY = None
    

    # setup for evaluation
    convert_inplace_relu_to_relu(model)
    set_requires_grad_True(model)
    model = model.eval()
    model = model.to(DEVICE)
    

    # get a test data loader
    _, loader = get_loaders(model, dataset="cifar100", data_dir=DATA_DIR,
                            batch_size=BATCH_SIZE, num_train=NUM_IMGS, num_val=NUM_IMGS,
                            args=args,
                           )
    
    # Output class destinate how to calculate LEE
    # To recognize the class, one sample output is needed. 
    def get_output_type(model, loader):
        x, t = loader.__iter__().__next__()
        x = x.to(DEVICE)
        return type(model(x)) 
        
    output_type = get_output_type(model, loader)
    
    # prepare a model
    lee_model = copy.deepcopy(model)
    handles = lee.apply_hooks(lee_model, BATCH_SIZE, TRANSFORM)
    
    # prepare a LEE calculation function
    calc_errors = partial(lee.compute_equivariance_attribution,
                          lee_model,
                          num_probes=NUM_PROBES,
                          top_layer=TOP_LAYER,
                          output_type=output_type,
                          output_key=OUTPUT_KEY,
                          normalize=NORMALIZE
                         )
    
    # Calculation LEE
    lee_metrics = get_layerwise(calc_errors, loader, 
                                MODEL_NAME, NUM_IMGS, NUM_PROBES, DEVICE)
    
    # write a data file
    os.makedirs(OUT_DIR, exist_ok=True) # root directory
    
    OUT_DIR = os.path.join(OUT_DIR, f"lee_{TRANSFORM}")
    os.makedirs(OUT_DIR, exist_ok=True) # subdirectory

    filename = f'{MODEL_NAME}.csv' if UNTRAINED is False else f'{MODEL_NAME}_untrained.csv'
    OUT_FILE = os.path.join(OUT_DIR, filename)
    lee_metrics.to_csv(OUT_FILE)
    
    
    # TODO:
    # handling of other metrics
    #
    #other_metrics_transforms = ["integer_translation","translation","rotation"]
    #if (not args.use_lee) and (args.transform in other_metrics_transforms):
    #    other_metrics_model = copy.deepcopy(model)
    #    other_metrics.apply_hooks(other_metrics_model)
    #    func = partial(other_metrics.compute_equivariance_attribution, args.transform)
    #    other_metrics_results = get_layerwise(
    #        args, other_metrics_model, loader, func=func
    #    )
    #
    #    other_metrics_output_dir = os.path.join(args.output_dir, "stylegan3_" + args.transform)
    #    os.makedirs(other_metrics_output_dir, exist_ok=True)
    #    results_fn = args.model_name + "_norm_sqrt" + ".csv"
    #    other_metrics_results.to_csv(os.path.join(other_metrics_output_dir, results_fn))


def get_args_parser():

    parser = argparse.ArgumentParser(description="Layerweise LEE calculation")
    parser.add_argument(
        "--gpu_id", "-g", default=0, type=int,
        help="gpu id. default=0"
    )
    parser.add_argument(
        "--model_name", "-m", metavar='MODEL NAME', default="resnet50",
        help="Model name for timm library."
    )
    parser.add_argument(
        "--untrained", "-x", action='store_true',
        help="flag for loading untrained models. If this argument is present, untrained models are loaded."
    )
    parser.add_argument(
        "--chkpt_list", "-c", metavar='CHECKPOINT LIST', default="/home/chanseok-lim/raid/models/additional_models.json",
        help="Directory of a json file to look up model checkpoint or model configuration information"
    )
    parser.add_argument(
        "--data_dir", "-d", metavar='DATA DIR', type=str,
        default='/home/chanseok-lim/raid/ILSVRC2012/',
        help='Root directory of dataset to use'
    )
    parser.add_argument(
        "--batch_size", "-b", metavar='BATCH SIZE', type=int, default=1, 
        help="Batch size. default=1"
    )
    parser.add_argument(
        "--num_imgs", "-i", type=int, default=100, 
        help="Number of images to evaluate over. default=100"
    )
    parser.add_argument(
        "--num_probes", "-j", type=int, default=100,
        help="Number of probes, i,e., Number of evaluations to one image with peturbation by rondom vectors.\
        default=100",
    )
    parser.add_argument(
        "--transform", "-t", default="translation",
        choices=['translation', 'rotation', 'hyper_rotation', 'scale', 'saturate'],
        help="Transformation type for calculation of lie deriv",
    )
    parser.add_argument(
        "--normalize", "-z", action='store_true',
        help="flag for normalization of lee."
    )
    #parser.add_argument(
    #    "--use_lee", action='store_true',
    #    help="Use LEE (rather than metric not in limit)"
    #)
    parser.add_argument(
        '--out_dir', "-o", metavar='OUT DIR', default='equivariance_metrics_cnns',
        help='Directory to write a data file',
    )
    
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    