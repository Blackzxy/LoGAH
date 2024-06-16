
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sys
import torchvision
import torch.utils
import os
import math
import pickle
import numpy as np
import time
from tqdm import tqdm
# from ppuda.deepnets1m.graph import GraphBatch
from ppuda.utils import capacity
# from ppuda.ghn.nn import GHN

from ghn_lora.ghn3.nn import GHN3
#from ghn_lora.ghn3.nn import from_pretrained, get_metadata
from ghn3.ghn3.nn import from_pretrained
from ghn3.graph import GraphBatch
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from torchvision.models.vision_transformer import _vision_transformer, ViT_B_16_Weights, ViT_L_16_Weights


# all_torch_models = (torchvision.models.vision_transformer.__all__ +
#          torchvision.models.swin_transformer.__all__ +
#          # ['maxvit_t'] +
#          torchvision.models.efficientnet.__all__ +
#          torchvision.models.convnext.__all__ +
#          torchvision.models.resnet.__all__ +
#          torchvision.models.regnet.__all__ +
#          ['alexnet', 'googlenet', 'inception_v3'] +
#          torchvision.models.vgg.__all__ +
#          torchvision.models.squeezenet.__all__ +
#          torchvision.models.densenet.__all__ +
#          torchvision.models.mobilenet.__all__ +
#          torchvision.models.mnasnet.__all__ +
#          torchvision.models.shufflenetv2.__all__)
layers = [12, 12, 24]
heads =  [6, 12, 16]
C = [384, 768, 1024]

all_torch_models = (#"ViT_B_16_Weights", 
                    #"ViT_L_16_Weights",
                    #"vit_b_16",
                    #"vit_l_16",
                    "vit_s_2",
                    )

for i, (l, h, c) in enumerate(zip(layers, heads, C)):
    kw_args = {
                    'patch_size': 16,
                    'num_layers': l,
                    'num_heads': h,
                    'hidden_dim': c,
                    'mlp_dim': int(c*4),
                    'num_classes': 1000,
                    'image_size': 224,
            }
    if c == 384:
        kw_args = {
                    'patch_size': 2,
                    'num_layers': l,
                    'num_heads': h,
                    'hidden_dim': c,
                    'mlp_dim': int(c*4),
                    'num_classes': 100,
                    'image_size': 32,
            }
        vit_s_c100 = _vision_transformer(
            weights=None,
            progress=True,
              **kw_args
        )

    elif c == 768:
        vit_b_image = _vision_transformer(
            weights=ViT_B_16_Weights.IMAGENET1K_V1,
            progress=True,
              **kw_args
        )
        vit_b_image_no_pretrained = _vision_transformer(
            weights=None,
            progress=True,
              **kw_args
        )
    else:
        vit_l_image = _vision_transformer(
            weights=ViT_L_16_Weights.IMAGENET1K_V1,
            progress=True,
              **kw_args
        )
        vit_l_image_no_pretrained = _vision_transformer(
            weights=None,
            progress=True,
              **kw_args
        )


if len(sys.argv) > 1:
    #ghn = GHN3.load(sys.argv[1], debug_level=0)
    ghn, config, state_dict = from_pretrained(sys.argv[1], debug_level=0)
else:
    ghn = None

pretrained = ghn is None
print('pretrained: ', pretrained)

if pretrained:
    all_weights = []
    for w in all_torch_models:
        if w[0].isupper() and w.endswith('_Weights'):
            all_weights.append(w)
    all_weights = list(set(all_weights))

# shapes = {}
# for arch in tqdm(all_torch_models()):
#     try:
#         model = eval('torchvision.models.{}(pretrained=True)'.format(arch))
#     except Exception as e:
#         weights = None
#         for w in all_weights:
#             if w.lower().startswith(arch):
#                 weights = eval('torchvision.models.{}.IMAGENET1K_V1'.format(w))
#                 break
#         model = eval('torchvision.models.{}(weights=weights)'.format(arch))
#     for p in model.parameters():
#         s = tuple(p.shape)
#         if s not in shapes:
#             shapes[s] = 0
#         shapes[s] += 1
#
#     model = ghn(model)
#
# shapes = sorted(shapes.items(), key=lambda x: x[1])
#
# for s in shapes[::-1]:
#     print(s)

#features = {(64, 3, 7, 7): [], (256, 256, 3, 3): [], (1024, 1024, 1, 1): []}  # , (768, 768): []
features  ={(384, 384): [], (1536, 384): []}
for arch in tqdm(all_torch_models):
    if arch[0].isupper():
        continue

    print(arch)
    # if arch in ['efficientnet_v2_l', 'vit_h_14', 'regnet_y_128gf']:
    #     print('skipping as not supported by GHNs for now')
    #     continue
    weights, model = None, None
    skip = False
    try:
        if pretrained:
            #model = eval('torchvision.models.{}(pretrained=True)'.format(arch))
            if arch == 'vit_s_2':
                model = vit_s_c100
                model.expected_input_sz = 32

            elif arch == "vit_b_16":
                model = vit_b_image
            else:
                model = vit_l_image
        else:
            # model = eval('torchvision.models.{}()'.format(arch))
            if arch == 'vit_s_2':
                model = vit_s_c100
                model.expected_input_sz = 32

            elif arch == "vit_b_16":
                model = vit_b_image_no_pretrained
            else:
                model = vit_l_image_no_pretrained

    except:
        if pretrained:
            for w in all_weights:
                if w.lower().startswith(arch):
                    print(arch, w)
                    try:
                        weights = eval('torchvision.models.{}.IMAGENET1K_V1'.format(w))
                    except AttributeError:
                        skip = True
                    break
            if skip:
                continue
            model = eval('torchvision.models.{}(weights=weights)'.format(arch))
        else:
            raise

    # if capacity(model)[1] > 200 * 10**6:
    #     print('the model is too big, skipping', capacity(model)[1])
    #     continue

    if not pretrained:
        if isinstance(model, torchvision.models.Inception3):
            model.expected_image_sz = 299
            model.expected_input_sz = 299
        

        model = ghn(model)

    for p in model.parameters():
        s = tuple(p.shape)
        print(s)
        if s in features:
            features[s].append(p.data.cpu().numpy())

    # break

# with open('ghn2_filters.pkl', 'wb') as f:
#     pickle.dump(features, f)


def hungarian_cost(arr1, arr2):
    cost_matrix = np.abs(pairwise_distances(arr1, arr2, metric='cosine'))
    # print(arr1.shape, arr2.shape, cost_matrix.shape)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # c = cost_matrix[row_ind, col_ind].sum()
    # print(c)
    return arr2[col_ind]


for s in features:
    if len(features[s]) == 0:
        continue
    x = np.stack(features[s])
    # print(s, x.shape)
    x = x.reshape((x.shape[0], x.shape[1], np.prod(x.shape[2:])))
    d = np.zeros((len(x), len(x)))
    for i, a in enumerate(x):
        # print(i, len(x), a.shape)
        for j, b in enumerate(x):
            if j >= i:
                # print(a.shape, b.shape)
                # p = pairwise_distances(a.reshape(1, -1), b.reshape(1, -1), metric='cosine')[0, 0]
                p = pairwise_distances(a.reshape(1, -1), hungarian_cost(a, b).reshape(1, -1), metric='cosine')[0, 0]
                # print(p1, p)
                d[j, i] = d[i, j] = p

    # d = np.abs(pairwise_distances(x, x, metric='cosine'))
    print(s, x.shape, np.abs(d).mean())
