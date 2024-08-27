# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluates a GHN on LLMs.

Example

    # Running for Llama:
    python eval_ghn_for_llm.py --ckpt checkpoint_epoch90.pt --split meta-llama/Meta-Llama-3.1-8B --hf_token $token --debug 1

    # Running for GPT2:
    python eval_ghn_for_llm.py --ckpt checkpoint_epoch90.pt --split gpt2 --hf_token $token --debug 1

    where $token is your Hugging Face authentication token.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import time
import argparse
from huggingface_hub import login
from ppuda.config import init_config
from ghn3.nn import from_pretrained, get_metadata
from ghn3.graph import Graph_LLM, GraphBatch
from ghn3.utils import capacity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


parser = argparse.ArgumentParser(description='Evaluation of GHNs')
parser.add_argument('--save_ckpt', type=str, default=None,
                    help='checkpoint path to save the model with predicted parameters')
parser.add_argument('--hf_token', type=str, help='Hugging Face authentication token')
args = init_config(mode='eval', parser=parser, debug=0, split='torch')
try:
    login(token=args.hf_token, add_to_git_credential=True)  # need to login to download some models like Llama
except Exception as e:
    print(e)

ghn, config, state_dict = from_pretrained(args.ckpt, debug_level=args.debug)  # get a pretrained GHN
ghn = ghn.to(args.device)
ghn.eval()  # should be a little bit more efficient in the eval mode

norms = get_metadata(args.ckpt, attr='paramnorm')  # load meta-data for sanity checks

start_all = time.time()

# args.split can be meta-llama/Llama-2-7b-hf, meta-llama/Meta-Llama-3.1-8B, etc
tokenizer = AutoTokenizer.from_pretrained(args.split)
config = AutoConfig.from_pretrained(args.split)

if args.split.startswith('meta-llama'):
    # some config for a smaller model
    n_embd = 1024
    n_layer = 12
    n_head = 16
    config.hidden_size = n_embd
    config.intermediate_size = n_embd * 4
    config.num_hidden_layers = n_layer
    config.num_attention_heads = n_head
    config.num_key_value_heads = int(n_head/2)
    # config.tie_word_embeddings = True

model = AutoModelForCausalLM.from_config(config)

print(config)
print('\n{}/{}: {} with {:.2f}M parameters'.format(1,
                                                   1,
                                                   args.split.upper(),
                                                   capacity(model)[1]), end='...')
if args.device != 'cpu':
    torch.cuda.synchronize()
start = time.time()

with torch.no_grad():  # to improve efficiency
    graph = Graph_LLM(model, tokenizer, ve_cutoff=250, dense=True)
    name = '{}_{}'.format(args.split.split('/')[-1], 'tie' if model.config.tie_word_embeddings else 'no-tie')
    print('figure name', name)
    graph.visualize(figsize=(15, 15), with_labels=True, detailed_labels=True,
                    font_size=10, figname=name)  # save the pdf figure of the computation graph

    print('model param norm', sum([p.norm() for p in model.parameters()]),
          'num_params', capacity(model)[1],
          flush=True)
    for n, p in model.named_parameters():
        if n.find('lm_head') >= 0 or n.find('embed_tokens.weight') >= 0 or n.find('.wte') >= 0:
            print(n, p.shape, p.numel(), p.norm(), p.data_ptr())

    model = ghn(model.to(args.device), GraphBatch([graph], dense=True).to_device(args.device))
    print('model param norm', sum([p.norm() for p in model.parameters()]),
          'num_params', capacity(model)[1],
          flush=True)

    for n, p in model.named_parameters():
        if n.find('lm_head') >= 0 or n.find('embed_tokens.weight') >= 0 or n.find('.wte') >= 0:
            print(n, p.shape, p.numel(), p.norm(), p.data_ptr())

    if args.save_ckpt is not None:
        torch.save({'state_dict': model.state_dict()}, args.save_ckpt)
        print('\nsaved the model with predicted parameters to {}\n'.format(args.save_ckpt))

