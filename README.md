# LoGAH
This is for the paper **LoGAH: Predicting 774-Million-Parameter Transformers using Graph HyperNetworks with $\frac{1}{100}$ Parameters**

## Requirements
Please refer to [Parameter Prediction for Unseen Deep Architectures](https://github.com/facebookresearch/ppuda/tree/main) for preparing the running environments, and also git clone the [GHN-3](https://github.com/SamsungSAILMontreal/ghn3/tree/main) since our codes are based on it.


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

## LoGAH
We provide the LoGAH in `nn.py` in `ghn3` folder, where we introduce the low-rank decoder, as well as the modified `trainer.py` for our paper. We also modify the `config.py` in `ppuda` by adding the hyperparameter such as `--lora_r` used in LoGAH.

## Training
You can follow the instructions in GHN-3 to run the training on multiple GPUs.
