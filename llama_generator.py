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
from ghn3.graph import Graph_LLM, GraphBatch
from ppuda.utils.utils import capacity, set_seed

from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from huggingface_hub import login
hf_token = "XXX"
login(token=hf_token, add_to_git_credential=True)  # need to login to download some models like Llama

#def main():
try:
    split = 'train'
    N = int(sys.argv[1])
    data_dir = sys.argv[2]
except Exception as e:
    print('\nExample of usage: python llama_generator.py 1000 ./data\n', e)
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
h5_file = join(data_dir, 'Llama1K_%s.pkl' % split)

graphs = []
params = []
model_id = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)


def get_var():
    
    var = model(**tokenizer("Hello, my dog is cute", return_tensors="pt")).logits
    return var


while len(graphs) < N:
    n_layer = np.random.randint(3, 10)
    if n_layer > 5:
        n_heads = np.random.choice([4])
        ## make sure (n_embd/n_heads)%2 == 0
        x_min = 2**4
        x_max = 2**5
        x_ = np.random.choice(np.arange(x_min, x_max+1, 2))
        n_embd = x_ * n_heads
    elif n_layer > 3:
        n_heads = np.random.choice([6])
        x_min = 2**4
        x_max = 2**5
        x_ = np.random.choice(np.arange(x_min, x_max+1, 2))
        n_embd = x_ * n_heads
    else:
        n_heads = np.random.choice([8])
        x_min = 2**3
        x_max = 2**4
        x_ = np.random.choice(np.arange(x_min, x_max+1, 2))
        n_embd = x_ * n_heads
    
    config.hidden_size = int(n_embd)
    config.intermediate_size = int(n_embd * 4)
    config.num_hidden_layers = int(n_layer)
    config.num_attention_heads = int(n_heads)
    config.num_key_value_heads = int(n_heads)

    net_args = {'n_embd': n_embd, 'n_layer': n_layer, 'n_head': n_heads}
    print(net_args, flush=True)
    print(config, flush=True)
    #model = LlamaForCausalLM(config)
    model = AutoModelForCausalLM.from_config(config)

    
    
    # model.get_var = get_var
    #print(model.get_var)
    n = capacity(model)[1]
    if n > max_params:
        print('too large archi: %.2f M params \n' % (n / 1e6), flush=True)
        continue
    
    params.append(n/1e6)
    graph = Graph_LLM(model, tokenizer, ve_cutoff=250, dense=True)
    graph.net_args = {'n_embd': n_embd, 'n_layer': n_layer, 'n_head': n_heads}
    graph.config = config
    graphs.append(graph)
    print(len(graphs), '%.3f M params' % (n / 1e6), flush=True)

# with open(h5_file, 'wb') as f:
#     pickle.dump(graphs, f)

print('saved to %s' % h5_file)
print('params: %.3f +- %.3f M' % (np.mean(params), np.std(params)), flush=True)
print('\n done')



# if __name__ == '__main__':
#     main()
