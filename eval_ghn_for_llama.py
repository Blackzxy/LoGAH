# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluates a GHN on one or all PyTorch models on ImageNet.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

Example

    # Evaluating on all PyTorch models:
    python eval_ghn.py -d imagenet -D $SLURM_TMPDIR --ckpt ghn3xlm16.pt --split torch

    # Evaluating a single model like ResNet-50:
    python eval_ghn.py -d imagenet -D $SLURM_TMPDIR --ckpt ghn3xlm16.pt --arch resnet50 --split torch

    # Evaluating on all DeepNets1 models in the predefined split:
    python eval_ghn.py --ckpt ./checkpoints/ghn3tm8-c10-e833cce-1111/checkpoint.pt --split predefined
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torchvision.models as models
import time
import argparse
import inspect
import ppuda.ema as ema
from ppuda.config import init_config
from ppuda.utils import infer, AvgrageMeter, adjust_net
from ppuda.vision.loader import image_loader
#from ghn3.nn2 import from_pretrained, get_metadata
from ghn3.nn import from_pretrained, get_metadata
from ghn3.graph import Graph_GPT, GraphBatch
from ghn3.gpt2_1k import GPT2_1KDDP
#from ghn3 import from_pretrained, get_metadata, DeepNets1MDDP
from torchvision.models.vision_transformer import _vision_transformer
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3-8B-hf")
model = AutoModelForCausalLM.from_pretrained("Meta-Llama-3-8B-hf")


parser = argparse.ArgumentParser(description='Evaluation of GHNs')
parser.add_argument('--save_ckpt', type=str, default=None,
                    help='checkpoint path to save the model with predicted parameters')
args = init_config(mode='eval', parser=parser, debug=0, split='torch')

ghn, config, state_dict = from_pretrained(args.ckpt, debug_level=args.debug) # get a pretrained GHN
ghn = ghn.to(args.device)
ghn.eval()  # should be a little bit more efficient in the eval mode




norms = get_metadata(args.ckpt, attr='paramnorm')  # load meta-data for sanity checks

is_torch = args.split == 'torch'
if is_torch:
    # Enumerate all PyTorch models of ImageNet classification
    # Should be >= 74 models in torchvision>=0.13.1
    models_queue = []
    models_queue.append('llama')
    # for m in dir(models):
    #     if m[0].isupper() or m.startswith('_') or m.startswith('get') or m == 'list_models' or \
    #             not inspect.isfunction(eval('models.%s' % m)):
    #         continue

    #     if args.arch is not None and m == args.arch:
    #         models_queue = [m]
    #         break

    #     if norms is not None and m not in norms:
    #         print('=== %s was not in PyTorch at the moment of GHN-3 evaluation, so skipping it in this notebook ==='
    #               % m.upper())
    #         continue  # skip for consistency with the paper

    #     models_queue.append(m)
    print('\n%d PyTorch models found. Predicting parameters for all...' % len(models_queue))



start_all = time.time()
norms_matched = []

for m_ind, m in enumerate(models_queue):

    try:
        # Predict parameters


        n_params = sum([p.numel() for p in model.parameters()]) / 10 ** 6
        print('\n{}/{}: {} with {:.2f}M parameters'.format(m_ind + 1,
                                                           len(models_queue),
                                                           m.upper(),
                                                           n_params), end='...')
        if args.device != 'cpu':
            torch.cuda.synchronize()
        start = time.time()

        # if is_torch:
        #     model = adjust_net(model, large_input=False)  # adjust the model for small images such as 32x32 in CIFAR-10

        with torch.no_grad():  # to improve efficiency
            graph = Graph_GPT(model, ve_cutoff=250, dense=True)
            model = ghn(model.to(args.device), GraphBatch([graph], dense=True).to_device(args.device))
            if args.save_ckpt is not None:
                torch.save({'state_dict': model.state_dict()}, args.save_ckpt)
                print('\nsaved the model with predicted parameters to {}\n'.format(args.save_ckpt))

           

    except Exception as e:
        print('ERROR for model %s: %s' % (m, e))
