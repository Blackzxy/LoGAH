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
from loader import DeepNets1M, NetBatchSampler, MAX_NODES_BATCH
from .graph import Graph, GraphBatch
from .utils import log
from .ddp_utils import is_ddp
from .ops import NetworkLight

print("################## Load ViT Dataset ##################")
class DeepNets1MDDP(DeepNets1M):
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
            deepnets_file = 'ViTs1K_%s.pkl' % (split if split in ['train', 'search'] else 'eval')
            h5_file = os.path.join(nets_dir, deepnets_file)
            if not os.path.exists(h5_file):
                kwargs['nets_dir'] = './data'
            log('ViT1mDDP nets_dir set to %s as ViTs1m files not found at %s' % (kwargs['nets_dir'], nets_dir))

        super(DeepNets1MDDP, self).__init__(**kwargs)
        self.wider_nets = wider_nets
        self.dense = dense
        self.debug = debug


    @staticmethod
    def loader(meta_batch_size=1, dense=True, **kwargs):
        nets = DeepNets1MDDP(dense=dense, **kwargs)
        sampler = NetBatchSamplerDDP(nets, meta_batch_size) if nets.is_train else None
        n_w = (0 if meta_batch_size <= 1 else min(8, max(4, meta_batch_size // 2))) if nets.is_train else 0
        log('num workers', n_w)
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=sampler,
                                             batch_size=1,
                                             pin_memory=True,
                                             collate_fn=partial(GraphBatch, dense=dense),
                                             num_workers=0)
        return (loader, sampler) if nets.is_train else loader  # need to return sampler for distributed training

    def __getitem__(self, idx):

        # if self.h5_data is None:  # A separate fd is opened for each worker process
        #     self.h5_data = h5py.File(self.h5_file, mode='r')
        if self.pk_data is None:
            with open(self.pk_file, 'rb') as f:
                self.pk_data = pickle.load(f)
        model = self.pk_data[idx][0]
        graph = self.pk_data[idx][1]

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
    def __iter__(self):
        epoch = 0
        while True:  # infinite sampler
            if is_ddp():
                log(f'shuffle train loader: set seed to {self.sampler.seed}, epoch to {epoch}')
                self.sampler.set_epoch(epoch)
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    if self.check_batch(batch):
                        yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.check_batch(batch):
                    yield batch
            epoch += 1
