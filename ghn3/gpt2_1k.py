# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loaders for DeepNets-1M supporting distributed training.

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch.utils.data
import networkx as nx
import h5py
import os
import pickle  
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from ppuda.utils import rand_choice
from genotypes import from_dict, PRIMITIVES_DEEPNETS1M
from ghn_lora.loader import GPT2_1K, NetBatchSampler, MAX_NODES_BATCH
from .graph import Graph, GraphBatch
from .utils import log
from .ddp_utils import is_ddp
from .ops import NetworkLight

print("################## Load GPT2 Dataset ##################")
class GPT2_1KDDP(GPT2_1K):
    r"""
    DeepNets1M loader supporting DDP.
    """

    def __init__(self,
                 dense=True,
                 wider_nets=True,
                 debug=False,
                 **kwargs):
        if 'nets_dir' in kwargs and kwargs['nets_dir'] != './data':
            # Reset to a local ./data folder if hdf5 is not found in nets_dir (handles some complicated cluster setups)
            nets_dir = kwargs['nets_dir']
            split = kwargs['split'] if 'split' in kwargs else 'train'
            # deepnets_file = 'ViTs1m_%s.hdf5' % (split if split in ['train', 'search'] else 'eval')
            deepnets_file = 'GPT21K_%s.pkl' % (split if split in ['train', 'search'] else 'eval')
            h5_file = os.path.join(nets_dir, deepnets_file)
            if not os.path.exists(h5_file):
                kwargs['nets_dir'] = './data'
            log('GPT21KDDP nets_dir set to %s as GPT21K files not found at %s' % (kwargs['nets_dir'], nets_dir))

        super(GPT2_1KDDP, self).__init__(**kwargs)
        self.wider_nets = wider_nets
        self.dense = dense
        self.debug = debug


    @staticmethod
    def loader(meta_batch_size=1, dense=True, **kwargs):
        nets = GPT2_1KDDP(dense=dense, **kwargs)
        sampler = NetBatchSamplerDDP(nets, meta_batch_size) if nets.is_train else None
        n_w = (0 if meta_batch_size <= 1 else min(8, max(4, meta_batch_size // 2))) if nets.is_train else 0
        log('num workers', n_w)
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=sampler,
                                             batch_size=1,
                                             pin_memory=False,
                                             collate_fn=partial(GraphBatch, dense=dense),
                                             num_workers=n_w)
        return (loader, sampler) if nets.is_train else loader  # need to return sampler for distributed training

    def __getitem__(self, idx):

        # if self.h5_data is None:  # A separate fd is opened for each worker process
        #     self.h5_data = h5py.File(self.h5_file, mode='r')
        if self.pk_data is None:
            with open(self.pk_file, 'rb') as f:
                self.pk_data = pickle.load(f)
        graph = self.pk_data[idx]

        return graph



class NetBatchSamplerDDP(NetBatchSampler):
    r"""
    NetBatchSampler that works with DistributedSampler.
    """

    def __init__(self, deepnets, meta_batch_size=1):
        super(NetBatchSampler, self).__init__(
            (DistributedSampler(deepnets) if is_ddp() else torch.utils.data.RandomSampler(deepnets))
            if deepnets.is_train
            else torch.utils.data.SequentialSampler(deepnets),
            meta_batch_size,
            drop_last=False)
        self.max_nodes_batch = int(
            MAX_NODES_BATCH / 8 * max(8, meta_batch_size)) if deepnets.is_train and meta_batch_size > 1 else None
        log('max_nodes_batch', self.max_nodes_batch, 'meta_batch_size', meta_batch_size)

    def check_batch(self, batch):
        return (self.max_nodes_batch is None or
                (self.sampler.dataset if is_ddp() else self.sampler.data_source).nodes[batch].sum() <=
                self.max_nodes_batch)
