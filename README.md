# LoGAH
This is for the paper [**LoGAH: Predicting 774-Million-Parameter Transformers using Graph HyperNetworks with $\frac{1}{100}$ Parameters**](https://arxiv.org/abs/2405.16287).

## Requirements
Please git clone [Parameter Prediction for Unseen Deep Architectures](https://github.com/facebookresearch/ppuda/tree/main) first and prepare the running environments, then git clone this repo under that folder.


## Dataset Generator
In `dataset_generator`, we provide the codes for generating ViTs-1K and GPTs-1K datasets. `gpt2_generator.py` and `vit_generator.py` is used for generating the dataset of GPTs-1K and ViTs-1K, respectively. Note that, when generating GPTs-1K datasets and training LoGAH on GPTs-1K, please modify the following code
```python
var = self.model(torch.randn(2, *self.expected_input_sz, device=device))
```
at  `traverse_graph` function  in `ghn3/graph.py` to
```python
var = self.model(**tokenizer("Hello, my dog is cute", return_tensors="pt")).logits
```
in order to support language models. We provided the modified `graph.py` in `ghn3` folder for your convenience.

> Note that: It could be a large file when generating the original datasets, one solution is to generate a similar dataset with smaller models sizes, and in the corresponding `trainer.py` you randomly generate the original size of model for training. Refer to the commented code in `trainer.py` for more details.

## LoGAH
We provide the LoGAH in `nn.py` in `ghn3` folder, where we introduce the low-rank decoder, as well as the modified `trainer.py` for our paper. We also modify the `config.py` in `ppuda` by adding the hyperparameter such as `--lora_r` used in LoGAH. The trainer for GPT-1K datasets is shown in `trainer_gpt2.py`.

## Training LoGAH
You can follow the instructions in [GHN-3](https://github.com/SamsungSAILMontreal/ghn3/tree/main) to run the training on multiple GPUs. For training LoGAH on ViTs-1K, please use `train_ghn_ddp.py`; for training it on GPTs-1K, please use `train_ghn_gpt2.py`.

For example:
```python
python train_ghn_ddp.py -n -v 50 --ln --amp -m 1 --name ghn-logah-r90-hid128-m1-layers5-heads16-clip5 -d cifar100 --hid 128 --lora_r 90 --layers 5 --heads 16 --opt adamw --lr 0.3e-3 --wd 1e-2 --scheduler cosine-warmup --debug 0 --max_shape 2048 --lora
```
Or if you want to train on multiple GPUs, you can use the `torchrun` to do that:
```python
torchrun --standalone --nnodes=1 --nproc_per_node=2 ghn_lora/train_ghn_gpt2.py -n -v 50 --ln --amp -m 2  --name ghn-gpt2-lora-wiki103-r32-hid64-layers3-heads8-m2 -d wikitext --hid 64 --lora_r 32 --layers 3 --heads 8 --opt adamw --lr 0.3e-3 --wd 1e-2 --scheduler cosine-warmup --debug 0 --max_shape 2048 --lora --batch_size 6
```
If you encounter the hanging issue, please add the command `export NCCL_P2P_DISABLE=1` before running the experiments.

## Predict Parameters
For predicting the parameters for ViT and GPT-2, we provide `eval_ghn.py` and `eval_ghn_for_gpt2.py` respectively. For example:
```python
python eval_ghn.py -d cifar100 --ckpt checkpoints/ghn-c100-lora-r32-hid64-m8-layers3-heads8-clip5/checkpoint.pt --save checkpoints/ghn-c100-lora-r32-hid64-m8-layers3-heads8-clip5/c100_vit_epoch300_L24_H16_C1024_init.pt --split torch
```

## Training ViTs and GPTs
We provide the training scripts for ViT and GPT-2 respectively, `train_vit.py` and `train_gpt2.py`. 

For training ViT from our prediction, you can use the following command:
```python
python train_vit.py --split predefined --arch 0 --epochs 100 -d cifar100 --batch_size 32 --opt adamw --lr 0.04e-3 --wd 1e-2 --ckpt  checkpoints/ghn-c100-lora-r32-hid64-m8-layers3-heads8-clip5/c100_vit_epoch300_L24_H16_C1024_init.pt
```

For training GPT-2, we also ulitise the DeepSpeed:
```python
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed --master_port 12345 --include localhost:2,3,4,5,6,7 ghn_lora/train_gpt2.py --fp16 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 --learning_rate 3e-6 --weight_decay 1e-2 --warmup_steps 500 --preprocessing_num_workers 8  --num_train_epochs 100 --deepspeed ds_config_1gpu.json --per_device_train_batch_size 2 --per_device_eval_batch_size 2  --config_name gpt2-large --tokenizer_name gpt2-large --do_train --do_eval --output_dir ./wikitext103-GPTLarge
```
