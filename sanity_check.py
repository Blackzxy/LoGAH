import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torchvision.models as models
import time
import argparse
import inspect
import ppuda.ema as ema
from ppuda.config import init_config
from ppuda.utils import infer, AvgrageMeter, adjust_net
from ppuda.vision.loader import image_loader
#from ghn3.nn2 import from_pretrained, get_metadata
from ghn3.nn import from_pretrained, get_metadata, GHN3_GPT
from ghn3.graph import Graph_GPT, GraphBatch
from ghn3.gpt2_1k import GPT2_1KDDP
#from ghn3 import from_pretrained, get_metadata, DeepNets1MDDP
from torchvision.models.vision_transformer import _vision_transformer
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3-8B-hf")
model = AutoModelForCausalLM.from_pretrained("Meta-Llama-3-8B-hf")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LoGAH = GHN3_GPT(
    max_shape=(2048, 2048, 16, 16),
    num_classes=10,
    hid=64,
    heads=8,
    layers=3,
    is_ghn2=False,
    pretrained=False,
    lora=True,
    lora_r=32,
    max_ck_lora=int(16 * 2048),
    use_1d_decoder=False,
).to(device)
LoGAH.eval()


graph = Graph_GPT(model, ve_cutoff=250, dense=True,)
model = LoGAH(
    model.to(device),
    GraphBatch([graph], dense=True).to_device(device),
)

