import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ghn3.nn import GHN3 as GHN3_GPT
from ghn3.nn import GHN3 as GHN3
from torchvision.models.vision_transformer import _vision_transformer
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import login
hf_token = "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"
login(token=hf_token, add_to_git_credential=True)  # need to login to download some models like Llama

# params = []
# for i in tqdm(range(1000)):
#     layers = np.random.randint(3, 10)
#     if layers > 5:
#         dim_min = 128
#         dim_max = 256
#     elif layers > 3:
#         dim_min = 256
#         dim_max = 384
#     else:
#         dim_min = 384
#         dim_max = 512

#     hidden_dim = np.random.choice(np.arange(dim_min, dim_max+1, 32))
#     mlp_dim = hidden_dim * 4

#     if hidden_dim % 12 == 0:
#         heads = np.random.choice([3, 6, 12])
#     elif hidden_dim % 6 == 0:
#         heads = np.random.choice([3, 6])
#     elif hidden_dim % 3 == 0:
#         heads = 3
#     else:
#         heads = np.random.choice([4, 8])



#     net = _vision_transformer(
#         patch_size = 2,
#         num_layers = layers,
#         num_heads = heads,
#         hidden_dim = hidden_dim,
#         mlp_dim = mlp_dim,
#         num_classes = 10,
#         image_size = 32,
#         weights = None,
#         progress = False,
        
#     )
#     param = sum(p.numel() for p in net.parameters()) / 1e6
#     params.append(param)

# ## plot params
# plt.hist(params, bins=20, color='lightpink', edgecolor='black')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xlabel("params (M)", fontsize=14)
# plt.ylabel("count", fontsize=14)
# plt.title("ViTs-1K params distribution", fontsize=16)
# ## save to pdf and close
# plt.savefig("vit_params.pdf")
# plt.close()



#tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_id = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
params = []

# n_embd = 1024
# n_layer = 12
# n_heads = 16

# ## llama3-small: n_embd=512, n_layer=12, n_heads=8 -- 181.67M
# ## llama3-medium: n_embd=1024, n_layer=12, n_heads=16,num_key_value_heads=8  -- 451.43M
# ## llama3-large: n_embd=1280, n_layer=16, n_heads=20 -- 747.8M

# config.hidden_size = int(n_embd)
# config.intermediate_size = int(n_embd * 4)
# config.num_hidden_layers = int(n_layer)
# config.num_attention_heads = int(n_heads)
# config.num_key_value_heads = int(n_heads/2)

# model = AutoModelForCausalLM.from_config(config)

# param = sum(p.numel() for p in model.parameters()) / 1e6
# print(param)

n_layers_list = []
n_heads_list = []
for i in tqdm(range(300)):
    n_layer = np.random.randint(3, 10)

    ### GPT2
    # if n_layer > 5:
    #     dim_min = 72
    #     dim_max = 176
    # elif n_layer > 3:
    #     dim_min = 128
    #     dim_max = 176
    # else:
    #     dim_min = 176
    #     dim_max = 256
    
    # n_embd = np.random.choice(np.arange(dim_min, dim_max+1, 8))

    # if n_embd % 8 == 0:
    #     n_head = np.random.choice([8])
    # elif n_embd % 6 == 0:
    #     n_head = np.random.choice([6])
    # elif n_embd % 4 == 0:
    #     n_head = 4

    if n_layer > 7:
        dim_min = 64
        dim_max = 128
    elif n_layer > 5:
        dim_min = 128
        dim_max = 192
    else:
        dim_min = 160
        dim_max = 256
    
    n_embd = np.random.choice(np.arange(dim_min, dim_max+1, 8))

    if n_embd % 8 == 0 and (n_embd / 8) % 2 == 0:
        n_heads = 8
    elif n_embd % 6 == 0 and (n_embd / 6) % 2 == 0:
        n_heads = 6
    elif n_embd % 4 == 0 and (n_embd / 4) % 2 == 0:
        n_heads = 4
    else:
        n_heads = 2
    
    n_key_value_heads = np.random.choice([1, 2, int(n_heads/2), n_heads])
    
    # n_embd = np.random.choice(np.arange(dim_min, dim_max+1, 4))

    # if n_embd % 8 == 0:
    #     n_head = np.random.choice([8])
    # # elif n_embd % 6 == 0:
    # #     n_head = np.random.choice([6])
    # elif n_embd % 4 == 0:
    #     n_head = 4


    # config = GPT2Config(
    #     # bos_token_id=tokenizer.bos_token_id,
    #     # eos_token_id=tokenizer.eos_token_id,
    #     n_embd=int(n_embd),
    #     n_layer=int(n_layer),
    #     n_head=int(n_head),
    #     tie_word_embeddings=False,
    # )
    # model = GPT2LMHeadModel(config)

    config.hidden_size = int(n_embd)
    config.intermediate_size = int(n_embd * 4)
    config.num_hidden_layers = int(n_layer)
    config.num_attention_heads = int(n_heads)
    config.num_key_value_heads = int(n_key_value_heads)

    n_layers_list.append(n_layer)
    n_heads_list.append(n_heads)

    model = AutoModelForCausalLM.from_config(config)

    param = sum(p.numel() for p in model.parameters()) / 1e6
    #print(param)
    params.append(param)

## plot params
plt.hist(params, bins=20, color='skyblue', edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel("params (M)", fontsize=14)
plt.ylabel("count", fontsize=14)
plt.title("Llama-1K params distribution2", fontsize=16)
## save to pdf and close
plt.savefig("llama_params2.pdf")
plt.close()

## scatter plot between n_layers and n_heads
plt.scatter(n_layers_list, n_heads_list, color='lightcoral')
plt.xlabel("n_layers", fontsize=14)
plt.ylabel("n_heads", fontsize=14)
plt.title("n_layers vs n_heads", fontsize=16)
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.savefig("n_layers_vs_n_heads.pdf")
plt.close()

# layers = 12
# heads = 6
# C = 384
# mlp_dim = C * 4
# dset_name = 'cifar10'

# net_args = {'patch_size': 2,
#                     'num_layers': layers,
#                     'num_heads': heads,
#                     'hidden_dim': C,
#                     'mlp_dim': mlp_dim,
#                     'num_classes': 1000 if dset_name == 'imagenet' else 10,
#                     'image_size': 224 if dset_name == 'imagenet' else 32,
#                     }
# model = _vision_transformer(weights=None,progress=False, **net_args)
# print("params of vit: ", sum(p.numel() for p in model.parameters()) / 1e6)

# MAX_SHAPE = 2048 * 4
# HID=64
# HEADS=8
# LAYERS=3
# R=int(HID/2)

# ## Hid=64: max_shape=2048: 2.54 , max_shape=4096: 3.62,  max_shape=8192: 5.78

# lora_ghn = GHN_Lora(
#     max_shape=(MAX_SHAPE, MAX_SHAPE, 16, 16),
#     num_classes=10,
#     hid=HID,
#     heads=HEADS,
#     layers=LAYERS,
#     is_ghn2=False, 
#     pretrained=False, 
#     lora=True, 
#     lora_r=R,
#     max_ck_lora=16 * MAX_SHAPE,
#     use_1d_decoder=False
# )
# print("params of lora_ghn: ", sum(p.numel() for p in lora_ghn.parameters()) / 1e6)


# ghn3 = GHN3(
#     max_shape = (HID, HID, 16, 16),
#     num_classes = 10,
#     hid = HID,
#     heads=HEADS,
#     layers=LAYERS,
#     is_ghn2=False,
#     pretrained=False,
# )
# print("params of ghn3: ", sum(p.numel() for p in ghn3.parameters()) / 1e6)