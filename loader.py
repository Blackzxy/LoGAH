# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loaders for DeepNets-1M.

"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch.utils.data
import torchvision
import json
import h5py
import pickle
import os
from torchvision.models.vision_transformer import _vision_transformer
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

# from ppuda.vit1m import genotypes
from ppuda.utils import adjust_net, rand_choice
# from ppuda.vit1m.genotypes import from_dict, PRIMITIVES_DEEPNETS1M
# from ppuda.vit1m.ops import *
# from ppuda.vit1m.net import Network
# from ppuda.vit1m.graph import Graph, GraphBatch
from ghn_lora.ghn3.graph import Graph, GraphBatch


MAX_NODES_BATCH = 2200  # to fit larger meta batches into GPU memory (decreasing this number further may create a bias towards smaller architectures)

class DeepNets1M(torch.utils.data.Dataset):
    r"""
    Default args correspond to training a baseline GHN on CIFAR-10.
    """

    def __init__(self,
                 split='train',
                 nets_dir='./data',
                 virtual_edges=1,
                 num_ch=(128, 512),# changed
                 fc_dim=(64, 512),
                 num_nets=None,
                 arch=None,
                 large_images=False,
                 verbose=False):
        super(DeepNets1M, self).__init__()

        self.split = split
        assert self.split in ['train', 'val', 'test', 'search',
                              'wide', 'deep', 'dense', 'bnfree', 'predefined'],\
            ('invalid split', self.split)
        self.is_train = self.split == 'train'

        self.virtual_edges = virtual_edges
        assert self.virtual_edges >= 1, virtual_edges

        if self.is_train:
            # During training we will randomly sample values from this range
            self.num_ch = torch.arange(num_ch[0], num_ch[1] + 1, 16)
            self.fc_dim = torch.arange(fc_dim[0], fc_dim[1] + 1, 64)

        self.large_images = large_images  # this affects some network parameters
        self.verbose = verbose

        # Load one of the splits
        if self.verbose:
            print('\nloading %s nets...' % self.split.upper())

        if self.split == 'predefined':
            self.nets = self._get_predefined()
            self.n_all = len(self.nets)
            # self.nodes = torch.tensor([net.n_nodes for net in self.nets])
        else:
            self.pk_data = None
            self.pk_file = os.path.join(nets_dir, 'test-ViTs1K_%s.pkl' % (split if split in ['train', 'search'] else 'eval'))
            print(self.pk_file)
            assert os.path.exists(self.pk_file), ('%s not found' % self.pk_file)
            with open(self.pk_file, 'rb') as f:
                self.pk_data = pickle.load(f)
            self.n_all = len(self.pk_data)
            self.nets = [self.pk_data[i][1] for i in range(self.n_all)]
            self.net_args = [self.pk_data[i][0] for i in range(self.n_all)]
            self.nodes = torch.tensor([net_arg['num_nodes'] for net_arg in self.net_args])


        #     # Load meta data to convert dataset files to graphs later in the _init_graph function
        #     to_int_dict = lambda d: { int(k): v for k, v in d.items() }
        #     with open(self.pk.file.replace('.hdf5', '_meta.json'), 'r') as f:
        #         meta = json.load(f)[split]
        #         n_all = len(meta['nets'])
        #         self.nets = meta['nets'][:n_all if num_nets is None else num_nets]
        #         self.primitives_ext =  to_int_dict(meta['meta']['primitives_ext'])
        #         self.op_names_net = to_int_dict(meta['meta']['unique_op_names'])
        #     self.h5_idx = [ arch ] if arch is not None else None
        #     self.nodes = torch.tensor([net['num_nodes'] for net in self.nets])

        # if arch is not None:
        #     arch = int(arch)
        #     assert 0 <= arch < len(self.nets), \
        #         'architecture with index={} is not available in the {} split with {} architectures in total'.format(
        #             arch, split, len(self.nets))
        #     self.nets = [self.nets[arch]]
        #     self.nodes = torch.tensor([self.nodes[arch]])

        # if self.verbose:
        #     print('loaded {}/{} nets with {}-{} nodes (mean\u00B1std: {:.1f}\u00B1{:.1f})'.
        #           format(len(self.nets),n_all,
        #                  self.nodes.min().item(),
        #                  self.nodes.max().item(),
        #                  self.nodes.float().mean().item(),
        #                  self.nodes.float().std().item()))


    @staticmethod
    def loader(meta_batch_size=1, **kwargs):
        nets = DeepNets1M(**kwargs)
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=NetBatchSampler(nets, meta_batch_size) if nets.is_train else None,
                                             batch_size=1,
                                             pin_memory=True,
                                             collate_fn=GraphBatch,
                                             num_workers=0,
                                             #num_workers=2 if meta_batch_size <= 1 else min(8, meta_batch_size)
                                             )
        return iter(loader) if nets.is_train else loader


    def __len__(self):
        return self.n_all


    def __getitem__(self, idx):

        if self.split == 'predefined':
            graph =  self.nets[idx]
        
        else:
            if self.pk_data is None:
                with open(self.pk_file, 'rb') as f:
                    self.pk_data = pickle.load(f)
            model = self.pk_data[idx][0]
            graph = self.pk_data[idx][1]

            # if self.pk_data is None:  # A separate fd is opened for each worker process
            #     self.pk_data = h5py.File(self.pk.file, mode='r')

            

        return graph

    def _get_predefined(self):
        graphs = []
        num_classes = 1000 if self.large_images else 100
        patch_size = 16 # 2 for cifar, 16 for imagenet
        layers = 12
        heads = 12
        C = 768
        mlp_dim = int(C * 4)  
        kw_args = {
                'patch_size': patch_size,
                'num_layers': layers,
                'num_heads': heads,
                'hidden_dim': C,
                'mlp_dim': mlp_dim,
                'num_classes': num_classes,
                'image_size': 224 if self.large_images else 32,
        }
        model = adjust_net(_vision_transformer(weights=None,progress=False, **kw_args),large_input=self.large_images)
        graph = Graph(model, ve_cutoff=250, dense=True)
        graph.net_args = kw_args
        graphs.append(graph)
        return graphs

class GPT2_1K(torch.utils.data.Dataset):
    r"""
    Default args correspond to training a baseline GHN on CIFAR-10.
    """

    def __init__(self,
                 split='train',
                 nets_dir='./data',
                 virtual_edges=1,
                 num_ch=(128, 512),# changed
                 fc_dim=(64, 512),
                 verbose=False):
        super(GPT2_1K, self).__init__()

        self.split = split
        assert self.split in ['train', 'val', 'test', 'search',
                              'wide', 'deep', 'dense', 'bnfree', 'predefined'],\
            ('invalid split', self.split)
        self.is_train = self.split == 'train'

        self.virtual_edges = virtual_edges
        assert self.virtual_edges >= 1, virtual_edges

        if self.is_train:
            # During training we will randomly sample values from this range
            self.num_ch = torch.arange(num_ch[0], num_ch[1] + 1, 16)
            self.fc_dim = torch.arange(fc_dim[0], fc_dim[1] + 1, 64)

        
        self.verbose = verbose

        # Load one of the splits
        if self.verbose:
            print('\nloading %s nets...' % self.split.upper())

        if self.split == 'predefined':
            self.nets = self._get_predefined()
            self.n_all = len(self.nets)
            # self.nodes = torch.tensor([net.n_nodes for net in self.nets])
        else:
            self.pk_data = None
            self.pk_file = os.path.join(nets_dir, 'GPT21K_%s.pkl' % (split if split in ['train', 'search'] else 'eval'))
            print(self.pk_file)
            assert os.path.exists(self.pk_file), ('%s not found' % self.pk_file)
            with open(self.pk_file, 'rb') as f:
                self.pk_data = pickle.load(f)
            self.n_all = len(self.pk_data)
            self.nets = [self.pk_data[i] for i in range(self.n_all)] # only store the graph


    @staticmethod
    def loader(meta_batch_size=1, **kwargs):
        nets = GPT2_1K(**kwargs)
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=NetBatchSampler(nets, meta_batch_size) if nets.is_train else None,
                                             batch_size=1,
                                             pin_memory=False,
                                             collate_fn=GraphBatch,
                                             num_workers=2 if meta_batch_size <= 1 else min(8, meta_batch_size))
        return iter(loader) if nets.is_train else loader


    def __len__(self):
        return self.n_all


    def __getitem__(self, idx):

        if self.split == 'predefined':
            graph =  self.nets[idx]
        
        else:
            if self.pk_data is None:
                with open(self.pk_file, 'rb') as f:
                    self.pk_data = pickle.load(f)
            graph = self.pk_data[idx]

            # if self.pk_data is None:  # A separate fd is opened for each worker process
            #     self.pk_data = h5py.File(self.pk.file, mode='r')

            

        return graph

    def _get_predefined(self):
        graphs = []
        n_layer = 12
        n_head = 6
        n_embd = 384
        config = GPT2Config(
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_embd=int(n_embd),
        )
        kw_args = {'n_embd': n_embd, 'n_layer': n_layer, 'n_head': n_head}
        model = GPT2LMHeadModel(config)
        graph = Graph(model, ve_cutoff=250, dense=True)
        graph.net_args = kw_args
        graphs.append(graph)
        return graphs



class NetBatchSampler(torch.utils.data.BatchSampler):
    r"""
    Wrapper to sample batches of architectures.
    Allows for infinite sampling and filtering out batches not meeting certain conditions.
    """
    def __init__(self, deepnets, meta_batch_size=1):
        super(NetBatchSampler, self).__init__(
            torch.utils.data.RandomSampler(deepnets) if deepnets.is_train
            else torch.utils.data.SequentialSampler(deepnets),
            meta_batch_size,
            drop_last=False)
        self.max_nodes_batch = MAX_NODES_BATCH if deepnets.is_train else None

    def check_batch(self, batch):
        return (self.max_nodes_batch is None or
                self.sampler.data_source.nodes[batch].sum() <=
                self.max_nodes_batch)

    def __iter__(self):
        while True:  # infinite sampler
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