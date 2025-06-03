import os
import gc
import warnings
import numpy as np
import argparse
import pandas as pd
from functools import partial
import json

import torch
# Jacobian of efficient attention is not implemented
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn

import sys
sys.path.append('pytorch-image-models')
import timm

from lee.e2e_lee import get_equivariance_metrics as get_lee_metrics
from lee.e2e_other import get_equivariance_metrics as get_discrete_metrics
from lee.loader import get_loaders, eval_average_metrics_wstd



def numparams(model):
    return sum(p.numel() for p in model.parameters())


def get_metrics(key, loader, metrics_config):
    ## unpack the config
    model = metrics_config['model']
    model_type = metrics_config['model_type']
    output_type = metrics_config['output_type']
    output_key = metrics_config['output_key']
    model_name = metrics_config['model_name']
    max_mbs = metrics_config['max_mbs']

    lee_metrics = eval_average_metrics_wstd(
        loader, 
        partial(get_lee_metrics,
                model=model,
                model_type=model_type,
                output_type=output_type,
                output_key=output_key
        ),
        max_mbs=max_mbs,
    )
    discrete_metrics = eval_average_metrics_wstd(
        loader, 
        partial(get_discrete_metrics, 
                model=model,
                model_type=model_type,
                output_type=output_type,
                output_key=output_key
        ), 
        max_mbs=max_mbs,
    )
    metrics = pd.concat([lee_metrics, discrete_metrics], axis=1)

    metrics["dataset"] = key
    metrics["model"] = model_name
    metrics["params"] = numparams(model)

    return metrics


class Subnetwork(nn.Module):
    def __init__(self, model, top_layer):
        super().__init__()
        self.original_model = model
        self.top_layer = top_layer

        assert self.top_layer in dict(model.named_modules()), \
            f"the module named '{self.top_layer}' does not exist. \
            You can check the module names by calling model.named_modules()."

    def forward(self, x):
        # get the output of the top layer with register_forward_hook.
        # the hook is removed after the forward pass
        # to keep self.original_model original state.

        outputs = {}
        module = dict(self.original_model.named_modules())[self.top_layer]

        def hook(module, input, output):
            outputs[self.top_layer] = output

        handle = module.register_forward_hook(hook)
        
        _ = self.original_model(x)
        handle.remove()

        return outputs[self.top_layer]


def get_args_parser():

    parser = argparse.ArgumentParser(description='end-to-end evaluation of equivariance')
    parser.add_argument(
        "--gpu_id", "-g", default=0, type=int,
        help="gpu id. default=0"
    )
    parser.add_argument(
        "--model_name", "-m", metavar='MODEL NAME', default="resnet50",
        help="Model name for timm library."
    )
    parser.add_argument(
        "--chkpt_list", "-c", metavar='CHECKPOINT LIST', default="/home/chanseok-lim/raid/models/mymodels.json",
        help="Directory of a json file to look up model checkpoint or model configuration information"
    )
    parser.add_argument(
        "--backbone", "-w", action='store_true',
        help="if true, model is created loading weights on a backbone model registered in the timm library."
    )
    parser.add_argument(
        "--model_type", "-t", metavar='MODEL TYPE', default="classifier", 
        choices=['classifier', 'encoder', 'subnetwork'],
        help="Model type. classifier, encoder or subnetwork. default=classifier"
    )
    parser.add_argument(
        "--output_key", "-k", metavar='OUTPUT KEY', default=None,
        help="output key for models whose outputs are not simple torch.Tensor. \
        default=None"
    )
    parser.add_argument(
        "--subnetwork_top", "-s", default=None,
        help="a module name of the subnetwork. the name must be registered in model.named_modules()."
    )
    parser.add_argument(
        "--dataset", "-n", metavar='DATASET', type=str,
        default="imagenet",
        help='dataset name, e.g., imagenet, cifar-100 etc.'
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
        "--calculate_accuracy", "-a", action='store_true',
        help="flag for computing accuracy of the model."
    )
    parser.add_argument(
        '--out_dir', "-o", metavar='OUT DIR', default='equivariance_metrics_cnns',
        help='Directory to write a data file',
    )
    
    return parser


def main(args):
    warnings.simplefilter('ignore')

    # unpuck args
    DEVICE = torch.device(type='cuda', index=args.gpu_id)
    MODEL_NAME = args.model_name
    CHKPT_LIST = args.chkpt_list
    BACKBONE = args.backbone
    MODEL_TYPE = args.model_type
    OUTPUT_KEY = args.output_key
    SUBNETWORK_TOP = args.subnetwork_top
    DATASET = args.dataset
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_IMGS = args.num_imgs
    CALCULATE_ACCURACY = args.calculate_accuracy
    OUT_DIR = args.out_dir

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print(f'GPU ID: {DEVICE}' if torch.cuda.is_available() else 'CPU')
    print(f'MODEL_NAME: {MODEL_NAME}')

    # load additionally available models
    with open(CHKPT_LIST, 'r') as f:
        chkpt_list = json.load(f)

    # load a model
    if MODEL_NAME in chkpt_list:
        if BACKBONE:
            ## get a model configuration
            backborn = chkpt_list[MODEL_NAME]["backbone"]
            weights = chkpt_list[MODEL_NAME]["weights"]
            num_classes = chkpt_list[MODEL_NAME]["num_classes"]
        
            ## instantiate a model & load weights
            model = timm.create_model(backborn, pretrained=False, num_classes=num_classes)
            model.load_state_dict(torch.load(weights), strict=False)
            model.eval()

        else:
            ## setup for a model
            import importlib.util
            
            def execute_setup(setup_file):
                spec = importlib.util.spec_from_file_location("setup_module", setup_file)
                setup_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(setup_module)

            setup_file =  chkpt_list[MODEL_NAME]["setup_file"]
            if setup_file:
                execute_setup(setup_file)
            
            ## load an model
            path = chkpt_list[MODEL_NAME]["path"]
            model = torch.load(path, weights_only=False)
            model.eval()

    else:
        model = timm.create_model(MODEL_NAME, pretrained=True)
        model.eval()

    # to GPU
    if torch.cuda.is_available():
        model = model.to(DEVICE)

    # create train and test loaders
    train_loader, test_loader = get_loaders(
        model, 
        dataset=DATASET, 
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE, 
        num_train=NUM_IMGS, 
        num_val=NUM_IMGS,
        args=args,
        train_split='train',
        val_split='validation',
        )

    # if subnetwork is specified, generate the subnetowrk.
    # this process must be after create data loaders
    # because the subnetwork does not inherit attributions of the original model.
    if MODEL_TYPE == 'subnetwork':
        model = Subnetwork(model, SUBNETWORK_TOP)

    # get the output type
    # Output class destinate how to calculate LEE
    # To recognize the class, one sample output is needed. 
    def get_output_type(model, loader):
        x, _ = loader.__iter__().__next__()
        x = x.to(DEVICE)
        return type(model(x)) 
        
    output_type = get_output_type(model, train_loader)

    # evaluate the model
    metrics_config = {
        'model': model,
        'model_type': MODEL_TYPE,
        'output_type': output_type,
        'output_key': OUTPUT_KEY,
        'model_name': MODEL_NAME,
        'max_mbs': 400,
    }

    evaluated_metrics = []
    evaluated_metrics += [
        get_metrics(f"{DATASET}_train", train_loader, metrics_config),
        get_metrics(f"{DATASET}_test", test_loader, metrics_config),
    ]
    gc.collect()

    # _, cifar_test_loader = get_loaders(
    #     model,
    #     dataset="torch/cifar100",
    #     data_dir="/scratch/nvg7279/cifar",
    #     batch_size=1,
    #     num_train=args.num_imgs,
    #     num_val=args.num_imgs,
    #     args=args,
    #     train_split='train',
    #     val_split='validation',
    # )

    # evaluated_metrics += [get_metrics(args, "cifar100", cifar_test_loader, model, max_mbs=args.num_imgs)]
    # gc.collect()

    # _, retinopathy_loader = get_loaders(
    #     model,
    #     dataset="tfds/diabetic_retinopathy_detection",
    #     data_dir="/scratch/nvg7279/tfds",
    #     batch_size=1,
    #     num_train=1e8,
    #     num_val=1e8,
    #     args=args,
    #     train_split="train",
    #     val_split="train",
    # )

    # evaluated_metrics += [get_metrics(args, "retinopathy", retinopathy_loader, model, max_mbs=args.num_imgs)]
    # gc.collect()

    # _, histology_loader = get_loaders(
    #     model,
    #     dataset="tfds/colorectal_histology",
    #     data_dir="/scratch/nvg7279/tfds",
    #     batch_size=1,
    #     num_train=1e8,
    #     num_val=1e8,
    #     args=args,
    #     train_split="train",
    #     val_split="train",
    # )

    # evaluated_metrics += [get_metrics(args, "histology", histology_loader, model, max_mbs=args.num_imgs)]
    # gc.collect()

    df = pd.concat(evaluated_metrics)

    ## calculate accuracy
    if CALCULATE_ACCURACY:
        ## retrieve the original model
        if MODEL_TYPE == 'subnetwork':
            model = model.original_model

        def evaluate_prediction(model, x, t):
            x, t = x.to(DEVICE), t.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            score = (pred == t).cpu().float().data.numpy()
            return score if x.shape[0] > 1 else [score]
        eval = lambda x, t: evaluate_prediction(model, x, t)

        ## on tran data
        scores = [s for x, t in train_loader for s in eval(x, t)]
        condition = df['dataset'] == f"{DATASET}_train"
        df.loc[condition, 'acc'] = scores

        ## on tran data
        scores = [s for x, t in test_loader for s in eval(x, t)]
        condition = df['dataset'] == f"{DATASET}_test"
        df.loc[condition, 'acc'] = scores

    # save the results
    if MODEL_TYPE == 'subnetwork':
        model_full_name = '_'.join([MODEL_NAME, SUBNETWORK_TOP])
    else:
        model_full_name = MODEL_NAME
    file_name = f'{model_full_name}_e2e.csv'
    df.to_csv(os.path.join(OUT_DIR, file_name))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
