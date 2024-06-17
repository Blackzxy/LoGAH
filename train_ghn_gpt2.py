# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains a Graph HyperNetwork (GHN-3) on DeepNets-1M and ImageNet. DistributedDataParallel (DDP) training is
used if `torchrun` is used as shown below.
This script assumes the ImageNet dataset is already downloaded and set up as described in scripts/imagenet_setup.sh.

Example:

    # To train GHN-3-T/m8 on ImageNet (make sure to put the DeepNets-1M dataset in $SLURM_TMPDIR or in ./data) on
    # single GPU, automatic mixed precision:
    python train_ghn_ddp.py -d imagenet -D $SLURM_TMPDIR -n -v 50 --ln \
    -e 75 --opt adamw --lr 4e-4 --wd 1e-2 -b 128 --amp -m 8 --name ghn3tm8 --hid 64 --scheduler cosine-warmup

    # 4 GPUs (DDP), automatic mixed precision (as in the paper):
    export OMP_NUM_THREADS=8
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_ghn_ddp.py -d imagenet -D $SLURM_TMPDIR -n -v 50 --ln \
    -e 75 --opt adamw --lr 4e-4 --wd 1e-2 -b 128 --amp -m 8 --name ghn3tm8 --hid 64 --scheduler cosine-warmup

    # Sometimes, there can be mysterious errors due to DDP (depending on the pytorch/cuda version).
    # So it can be a good idea to wrap this command in a for loop to continue training in case of failure.

    # To train GHN-3-T/m8 on CIFAR-10:
    python train_ghn_ddp.py -n -v 50 --ln -m 8 --name ghn3tm8-c10 --hid 64 --layers 3 --opt adamw --lr 4e-4 --wd 1e-2 \
     --scheduler cosine-warmup --amp

    # Use eval_ghn.py to evaluate the trained GHN-3 model on ImageNet/CIFAR-10.

"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import torch.distributed as dist
from functools import partial
from itertools import chain
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
import transformers
import numpy as np
from transformers import AutoTokenizer, AutoConfig, default_data_collator, get_scheduler
from ppuda.config import init_config
import ppuda.ema as ema
# from ppuda.vision.loader import image_loader

from ghn3.gpt2_1k import GPT2_1KDDP
from ghn3.nn import GHN3_GPT
from ghn3.utils import log
from ghn3.trainer_gpt2 import Trainer
from ghn3.ddp_utils import setup_ddp, clean_ddp
import time
log = partial(log, flush=True)


def main():
    parser = argparse.ArgumentParser(description='GHN-3 training')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-2-raw-v1', help='dataset config name')
    parser.add_argument('--model_name', type=str, default='gpt2', help='model name')
    parser.add_argument('--use_slow_tokenizer', action='store_true', help='use slow tokenizer')
    parser.add_argument('--block_size', type=int, default=None, help='block size')
    parser.add_argument('--overwrite_cache', action='store_true', help='overwrite cache')
    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       layers=3,  # default number of layers in GHN-3
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)

    if hasattr(args, 'multigpu') and args.multigpu:
        raise NotImplementedError(
            'the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. '
            'nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel '
            '(https://github.com/pytorch/pytorch/issues/659360).'
            'Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. '
            'nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top).')
    

    log('loading the %s dataset...' % args.dataset.upper())
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=not args.use_slow_tokenizer)
    config = AutoConfig.from_pretrained(args.model_name)
    raw_datasets = load_dataset(args.dataset,args.dataset_config_name)
    # Preprocess the dataset
    # Tokenize all the texts
    column_names = raw_datasets['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > config.max_position_embeddings:
            args.block_size = min(1024, config.max_position_embeddings)
        block_size = min(args.block_size, tokenizer.model_max_length)
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    ## create dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, 
        batch_size=args.batch_size, 
        collate_fn=default_data_collator, 
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    train_queue = train_dataloader





    hid = args.hid
    s = 16
    ### You can change the max_shape here
    dmax_shape = 2048

    #default_max_shape = (hid * 2, hid * 2, s, s) if ghn2 else (hid, hid, s, s)
    default_max_shape = (dmax_shape, dmax_shape, s, s) if ghn2 else (dmax_shape, dmax_shape, s, s)
    log('current max_shape: {} {} default max_shape: {}'.format(args.max_shape,
                                                                '=' if args.max_shape == default_max_shape else '!=',
                                                                default_max_shape))

    config = {'max_shape': args.max_shape, 'hypernet': args.hypernet,
              'lora': args.lora, 'lora_r': args.lora_r, 'max_ck_lora': args.max_ck_lora, 'use_1d_decoder': args.use_1d_decoder,
              'decoder': args.decoder, 'weight_norm': args.weight_norm, 've': args.virtual_edges > 1,
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2}

    ghn = GHN3_GPT(**config, debug_level=args.debug).to(args.device)
    ema_helper = None
    ### Apply EMA ###
    if args.ema:
        ema_helper = ema.EMAHelper(mu = args.ema_rate)
        ema_helper.register(ghn)
    ### Apply EMA ###
        
    graphs_queue, sampler = GPT2_1KDDP.loader(
        meta_batch_size=args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
        dense=ghn.is_dense(),
        split=args.split,
        nets_dir=args.data_dir,
        verbose=ddp.rank == 0,
        debug=args.debug > 0,
    )


    trainer = Trainer(ghn,
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd, 'momentum': args.momentum},
                      scheduler='mstep' if args.scheduler is None else args.scheduler,
                      scheduler_args={'milestones': args.lr_steps, 'gamma': args.gamma},
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0 if ghn2 else 3e-5,
                      save_dir=args.save,
                      ckpt=args.ckpt,
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      compile_mode=args.compile,
                      ema=args.ema,
                      ema_helper=ema_helper,
                      )

    log('\nStarting training GHN with {} parameters!'.format(sum([p.numel() for p in ghn.parameters()])))
    if ddp.ddp:
        # make sure sample order is different for each seed
        sampler.sampler.seed = args.seed
        log(f'shuffle DeepNets1MDDP train loader: set seed to {args.seed}')
        # for each DeepNets1MDDP epoch, the graph loader will be shuffled inside the ghn3/deepnets1m.py

    graphs_queue = iter(graphs_queue)

    cnt = 0
    for epoch in range(trainer.start_epoch, args.epochs):
        if cnt>=1:
            break
        cnt+=1

        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        trainer.reset_metrics(epoch)

        for step, batch in enumerate(train_queue, start=trainer.start_step):
        #for step, (images, targets) in enumerate(train_queue, start=trainer.start_step):

            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break
           
            trainer.update(batch, graphs=next(graphs_queue))
            #trainer.update(images, targets, graphs=next(graphs_queue))
            trainer.log(step)

            if args.save:
                # save GHN checkpoint
                trainer.save(epoch, step, {'args': args, 'config': config}, interm_epoch=args.interm_epoch)

        trainer.scheduler_step()  # lr scheduler step

    ## save steps and losses
    # steps = np.array(steps)
    # losses = np.array(losses)
    # np.save(f'steps_{args.lora_r}_{args.hid}_{args.heads}_{args.layers}.npy', steps)
    # np.save(f'losses_{args.lora_r}_{args.hid}_{args.heads}_{args.layers}.npy', losses)


    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()


if __name__ == '__main__':
    main()