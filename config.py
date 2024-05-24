# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Code for the paper
"Boris Knyazev, Michal Drozdzal, Graham Taylor, Adriana Romero-Soriano.
Parameter Prediction for Unseen Deep Architectures. NeurIPS 2021."

This script contains the config file shared by different modules.

"""
#NOTE: This is the config file in the original PPUDA repo.

import argparse
import subprocess
import platform
import time
import os
import torch
import torchvision
import torch.backends.cudnn as cudnn
from ppuda.utils import set_seed, default_device


def init_config(mode='eval', parser=None, verbose=True, **kwargs):
    # kwargs can be used to pass default values for some arguments

    if verbose:
        print('\nEnvironment:')
    env = {}
    try:
        # print git commit to ease code reproducibility
        env['git commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        if verbose:
            print(e, flush=True)
        env['git commit'] = 'no git'

    env['hostname'] = platform.node()
    env['torch'] = torch.__version__
    env['torchvision'] = torchvision.__version__
    try:
        assert list(map(lambda x: float(x), env['torch'].split('.')[:2])) >= [1, 9]
    except:
        if verbose:
            print('WARNING: PyTorch version {} is used, but version >= 1.9 is '
                  'strongly recommended for this repo!'.format(env['torch']))

    env['cuda available'] = torch.cuda.is_available()
    env['cudnn enabled'] = cudnn.enabled
    env['cuda version'] = torch.version.cuda
    env['start time'] = time.strftime('%Y%m%d-%H%M%S')
    if verbose:
        for x, y in env.items():
            print('{:20s}: {}'.format(x[:20], y))

    if parser is None:
        # create parser or use the existing one if provided
        parser = argparse.ArgumentParser(description='Parameter Prediction for Unseen Deep Architectures')

    # Data args
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='image dataset: cifar10/imagenet/PennFudanPed.')
    args = parser.parse_known_args()[0]
    dataset = args.dataset.lower()
    is_imagenet_wiki = dataset.startswith('imagenet') or dataset.startswith('wikitext')
    is_detection = dataset == 'pennfudanped'

    parser.add_argument('-D', '--data_dir', type=str, default='./data',
                        help='where image dataset and DeepNets-1M are stored')
    parser.add_argument('--test_batch_size', type=int, default=1 if is_detection else 64,
                        help='image batch size for testing')

    # Generic args
    parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
    parser.add_argument('-w', '--num_workers', type=int, default=8 if is_imagenet_wiki else (4 if is_detection else 0),
                        help='number of cpu processes to use')
    parser.add_argument('--device', type=str, default=default_device(), help='device: cpu or cuda')
    parser.add_argument('--debug', type=int, default=kwargs.pop('debug', 1),
                        help='the level of details printed out, 0 is the minimal level.')
    parser.add_argument('-C', '--ckpt', type=str, default=None,
                        help='path to load the network/GHN parameters from')

    is_train_ghn = mode == 'train_ghn'
    is_train_net = mode == 'train_net'
    is_train = is_train_ghn or is_train_net
    is_eval = mode == 'eval'

    # Generic training args
    parser.add_argument('--split', type=str, default=kwargs.pop('split', 'train' if is_train_ghn else 'predefined'),
                        help='training/testing split of DeepNets-1M')
    parser.add_argument('-i', '--imsize', type=int, default=kwargs.pop('imsize', 224 if is_imagenet_wiki else 32),
                        help='image size used to train and eval models')

    if is_eval or is_train_net:
        parser.add_argument('--arch', type=str,
                            default=kwargs.pop('arch', 'DARTS' if is_train_net else None),
                            help='one of the architectures: string for the predefined genotypes such as DARTS; '
                                 'the architecture index from DeepNets-1M')
        parser.add_argument('--noise', action='store_true', help='add noise to validation/test images')
        if is_train_net:
            parser.add_argument('--pretrained', action='store_true', help='use pretrained torchvision.models')

    if is_train:

        # Predefine default arguments
        if is_train_ghn:
            batch_size = 256 if is_imagenet_wiki else 64
            epochs = 300
            lr = 0.6e-3
            wd = 1e-5
        else:
            if is_detection:
                batch_size = 2
                epochs = 10
                lr = 0.005
                wd = 0.0005
            else:
                batch_size = 128 if is_imagenet_wiki else 96
                epochs = 250 if is_imagenet_wiki else 600
                lr = 0.1 if is_imagenet_wiki else 0.0015
                wd = 3e-5 if is_imagenet_wiki else 3e-4

        parser.add_argument('-b', '--batch_size', type=int, default=batch_size, help='image batch size for training')
        parser.add_argument('-e', '--epochs', type=int, default=epochs, help='number of epochs to train')
        parser.add_argument('--opt', type=str, default='sgd' if is_train_net else 'adam', help='optimizer')
        parser.add_argument('--lr', type=float, default=lr, help='initial learning rate')
        parser.add_argument('--scheduler', type=str, default=kwargs.pop('scheduler', None), help='lr scheduler')
        parser.add_argument('--grad_clip', type=float, default=5, help='grad clip')
        parser.add_argument('-l', '--log_interval', type=int, default=10 if is_detection else 100,
                            help='number of training iterations when print intermediate results')
        parser.add_argument('-S', '--save_dir', type=str, default='./checkpoints',
                            help='where to put all trained data and stuff')
        parser.add_argument('--multigpu', action='store_true', help='train on all gpus available')
        parser.add_argument('--wd', type=float, default=wd, help='weight decay')
        parser.add_argument('--name', type=str, default='EXP', help='experiment name')
        parser.add_argument('--amp', action='store_true', help='use Automatic Mixed Precision')
        parser.add_argument('--lr_steps', type=str, default=kwargs.pop('lr_steps',
                                                                       '200,250' if is_train_ghn else '30,60'),
                            help='epochs when to decrease lr (default is for GHN-2 or ResNet training)')
        parser.add_argument('-g', '--gamma', type=float, default=0.1, help='learning rate decay factor')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        parser.add_argument('--layers', type=int, default=kwargs.pop(
            'layers', 1 if is_train_ghn else (14 if is_imagenet_wiki else 20)),
                            help='total number of layers in the network/GHN to be trained, default is for DARTS')

        if is_train_ghn:

            # GHN-specific args
            parser.add_argument('--ln', action='store_true', default=False, help='apply LayerNorm for node embeddings')
            parser.add_argument('-M', '--num_nets', type=int, default=10 ** 6,
                                help='number of training architectures')
            parser.add_argument('-v', '--virtual_edges', type=int, default=kwargs.pop('virtual_edges', 1),
                                help='the maximum shortest path distance to add in case of virtual edges '
                                     '(values <=1 correspond to the baseline and will not add virtual edges)')
            parser.add_argument('-n', '--weight_norm', action='store_true', default=False,
                                help='normalize predicted weights')
            parser.add_argument('-m', '--meta_batch_size', type=int, default=kwargs.pop('meta_batch_size', 1),
                                help='how many nets to sample per batch of images')
            parser.add_argument('--decoder', type=str, default='conv', help='decoder to predict final parameters')
            parser.add_argument('-H', '--hypernet', type=str, default='gatedgnn', help='message propagation network')
            parser.add_argument('--hid', type=int, default=kwargs.pop('hid', 32), help='number of hidden units')
            parser.add_argument('--max_shape', type=str, default=kwargs.pop('max_shape', None),
                                help='max shape "c_out[,c_in][,height][,width]" of the predicted parameters. '
                                     'If None, max_shape will be calculated based on shape_multiplier and the '
                                     '--dataset argument and  according to the hyperparameters in the GHN-2 papers.')
            parser.add_argument('--ema', action='store_true', default=False)
            parser.add_argument('--ema_rate', type=float, default=0.999)
            ### LoRA specific args ###
            parser.add_argument('--lora', action='store_true', default=True, help='use LoRA')
            parser.add_argument('--lora_r', type=float, default=128, help='percentage of r')
            parser.add_argument('--max_ck_lora', type=int, default=2048 * 16, help='max ck for LoRA')
            parser.add_argument('--use_1d-decoder', action='store_true', default=False, help='use 1d decoder')
            ### LoRA specific args ###

        else:
            parser.add_argument('--init_channels', type=int, default=48 if is_imagenet_wiki else 36,
                                help='num of init channels, default is for DARTS')
            parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
            parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
            parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
            parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
            parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
            parser.add_argument('--n_shots', type=int, default=None,
                                help='number of training images per class for fine-tuning experiments')
            parser.add_argument('--init', type=str, default='rand', help='init method')
            parser.add_argument('--layer', type=int, default=0, help='layer after each to add noise')
            parser.add_argument('--beta', type=float, default=kwargs.pop('beta', 0),
                                help='standard deviation of the Gaussian noise added to parameters')
            parser.add_argument('--val', action='store_true', default=False, help='evaluate on the validation set')

    args = parser.parse_args()

    if is_train:
        args.lr_steps = list(map(int, args.lr_steps.split(',')))

    if is_train_ghn:
        s = 16 # if is_imagenet_wiki else 11
        if args.max_shape is None:
            shape_multiplier = kwargs.pop('shape_multiplier', 2.0)  # default is hid * 2 in GHN-2
            args.max_shape = (int(args.hid * shape_multiplier), int(args.hid * shape_multiplier), s, s)
        else:
            c = list(map(int, args.max_shape.split(',')))
            if len(c) == 1:
                args.max_shape = (c[0], c[0], s, s)
            elif len(c) == 2:
                args.max_shape = (c[0], c[1], s, s)
            elif len(c) == 4:
                args.max_shape = tuple(c)
            else:
                raise NotImplementedError('max_shape must be a string of 1, 2 or 4 integers separated by commas. '
                                          'Example: --max_shape "64", --max_shape "64,64" or --max_shape "64,64,11,11"')

    def print_args(args, name):
        print('\n%s:' % name)
        args = vars(args)
        for x in sorted(args.keys()):
            y = args[x]
            print('{:20s}: {}'.format(x[:20], y))
        print('\n', flush=True)

    if verbose:
        print_args(args, 'Script Arguments ({} mode)'.format(mode))

    if hasattr(args, 'multigpu'):
        if args.multigpu:
            n_devices = torch.cuda.device_count()
            if n_devices > 1:
                assert args.device == 'cuda', ('multigpu can only be used together with device=cuda', args.device)
                if is_train_ghn:
                    assert args.meta_batch_size >= n_devices, \
                        'multigpu requires meta_batch_size ({}) to be >= number of cuda device ({})'.format(
                            args.meta_batch_size, n_devices)
                    assert args.meta_batch_size % n_devices == 0, \
                        'meta_batch_size ({}) must be a multiple of the number of cuda device ({})'.format(
                            args.meta_batch_size, n_devices)
                if verbose:
                    print('{} cuda devices are available for multigpu training\n'.format(n_devices))
            else:
                if verbose:
                    print('multigpu argument is ignored: > 1 cuda devices necessary, '
                          'while only {} cuda devices are available\n'.format(n_devices))
                args.multigpu = False

    set_seed(args.seed)

    if mode != 'eval':
        args.save = None
        if args.save_dir not in ['None', 'none', '']:
            args.save = os.path.join(args.save_dir, '{}-{}-{}'.format(args.name, env['git commit'], args.seed))
            if verbose:
                print('Experiment dir: {}\n'.format(args.save))
            if not os.path.exists(args.save):
                try:
                    os.makedirs(args.save)  # create dirs recursively if necessary
                except Exception as e:  # catch the error if another process has created the path meanwhile
                    print(e)

    return args