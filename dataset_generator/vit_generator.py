"""
Example:

    python vit_generator.py 10000 ./data

"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
import os
import numpy as np
from os.path import join
import sys
import pickle
import subprocess
import joblib
from ghn3.graph import Graph
from ppuda.utils.utils import capacity, set_seed
from torchvision.models.vision_transformer import _vision_transformer


def main():

    try:
        split = 'train'
        N = int(sys.argv[1])
        data_dir = sys.argv[2]
    except Exception as e:
        print('\nExample of usage: python vit_generator.py 1000 ./data\n', e)
        raise

    try:
        gitcommit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print('gitcommit:', gitcommit, flush=True)
    except Exception as e:
        print(e, flush=True)

    device = 'cpu'  # no much benefit of using cuda
    max_params = 5 * 10 ** 6

    print(split, N, data_dir, device, flush=True)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    set_seed(1)

    dset_name = 'cifar10'

    h5_file = join(data_dir, 'test-ViTs1K_%s.pkl' % split)

    graphs = []
    params = []

    while len(graphs) < N:

        layers = np.random.randint(3, 10)
        # C_max = 128 if layers > 6 else 256
        # C = np.random.choice(np.arange(32, C_max + 1, 32)) # previous:(32, C_max + 1, 32)
        # mlp_dim = int(C * (1 if C > 128 else 4))
        # heads = np.random.choice([2, 8])

        if layers > 5:
            dim_min = 128
            dim_max = 256
        elif layers > 3:
            dim_min = 256
            dim_max = 384
        else:
            dim_min = 384
            dim_max = 512
        
        hidden_dim = np.random.choice(np.arange(dim_min, dim_max+1, 32))
        mlp_dim = hidden_dim * 4

        if hidden_dim % 12 == 0:
            heads = np.random.choice([3, 6, 12])
        elif hidden_dim % 6 == 0:
            heads = np.random.choice([3, 6])
        elif hidden_dim % 3 == 0:
            heads = 3
        else:
            heads = np.random.choice([4, 8])

        net_args = {'patch_size': np.random.choice([16, 32]) if dset_name == 'imagenet' else np.random.choice([2, 4]),
                    'num_layers': layers,
                    'num_heads': heads,
                    'hidden_dim': hidden_dim,
                    'mlp_dim': mlp_dim,
                    'num_classes': 1000 if dset_name == 'imagenet' else 10,
                    'image_size': 224 if dset_name == 'imagenet' else 32,
                    }
        print(net_args, flush=True)
        model = _vision_transformer(weights=None,
                                    progress=False,
                                    **net_args).to(device)
        model.expected_input_sz = (3, 32, 32) if dset_name == 'cifar10' else (3, 224, 224)
        n = capacity(model)[1]
        # if n > max_params:
        #     print('too large architecture: %.2f M params \n' % (float(n) / 10 ** 6), flush=True)
        #     continue

        params.append(n / 10 ** 6)
        graph = Graph(model, ve_cutoff=250, dense=True, list_all_nodes=True)

        A = graph._Adj.cpu().numpy().astype(np.uint8)
        net_args['num_nodes'] = int(A.shape[0])

        graph.net_args = net_args
        graphs.append((net_args, graph))
        print(len(graphs), '%.3f M params' % (float(n) / 10 ** 6), flush=True)

    with open(h5_file, 'wb') as fp:
        pickle.dump(graphs, fp)
    print('saved to %s' % h5_file)
    print('params: %.3f +- %.3f (%.3f - %.3f) M' % (np.mean(params), np.std(params), np.min(params), np.max(params)),
          flush=True)

    print('\ndone')


if __name__ == '__main__':
    main()