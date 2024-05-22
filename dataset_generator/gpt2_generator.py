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
from ghn3.graph import Graph, GraphBatch
from ppuda.utils.utils import capacity, set_seed

from transformers import AutoTokenizer, AutoConfig
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer




#def main():
try:
    split = 'train'
    N = int(sys.argv[1])
    data_dir = sys.argv[2]
except Exception as e:
    print('\nExample of usage: python gpt2_generator.py 10000 ./data\n', e)
    raise

try:
    gitcommit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    print('gitcommit:', gitcommit, flush=True)
except Exception as e:
    print(e, flush=True)

device = 'cpu'
max_params = 50 * 10 ** 6

print(split, N, data_dir, device, flush=True)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

set_seed(1)

dset_name = 'wikitext-2'
h5_file = join(data_dir, 'GPT21K_%s.pkl' % split)

graphs = []
params = []
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def get_var():
    
    var = model(**tokenizer("Hello, my dog is cute", return_tensors="pt")).logits
    return var


while len(graphs) < N:
    n_layer = np.random.randint(3, 10)

    if n_layer > 5:
        dim_min = 72
        dim_max = 176
    elif n_layer > 3:
        dim_min = 128
        dim_max = 176
    else:
        dim_min = 176
        dim_max = 256
    
    n_embd = np.random.choice(np.arange(dim_min, dim_max+1, 8))

    if n_embd % 8 == 0:
        n_head = np.random.choice([8])
    elif n_embd % 6 == 0:
        n_head = np.random.choice([6])
    elif n_embd % 4 == 0:
        n_head = 4
    


    config = GPT2Config(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_embd=int(n_embd),
        n_layer=int(n_layer),
        n_head=int(n_head),
        tie_word_embeddings=False,
    )
    net_args = {'n_embd': n_embd, 'n_layer': n_layer, 'n_head': n_head}
    print(net_args, flush=True)
    model = GPT2LMHeadModel(config)

    
    
    # model.get_var = get_var
    #print(model.get_var)
    n = capacity(model)[1]
    if n > max_params:
        print('too large archi: %.2f M params \n' % (n / 1e6), flush=True)
        continue
    
    params.append(n/1e6)
    graph = Graph(model, reduce_graph=False)
    graph.net_args = {'n_embd': n_embd, 'n_layer': n_layer, 'n_head': n_head}
    graph.config = config
    graphs.append(graph)
    print(len(graphs), '%.3f M params' % (n / 1e6), flush=True)

with open(h5_file, 'wb') as f:
    pickle.dump(graphs, f)

print('saved to %s' % h5_file)
print('params: %.3f +- %.3f M' % (np.mean(params), np.std(params)), flush=True)
print('\n done')



# if __name__ == '__main__':
#     main()
