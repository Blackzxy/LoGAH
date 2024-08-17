# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper to train models.

"""


import os
import math
import torch
import torch.nn as nn
import numpy as np
import psutil
import traceback
import torch.distributed as dist
from functools import partial
from transformers import get_scheduler, AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, LambdaLR
from ppuda.utils import AvgrageMeter, accuracy, capacity, init
from ppuda.ghn.nn import GHN
from .graph import Graph_LLM, GraphBatch
from .utils import Logger, print_grads, log
from .ddp_utils import is_ddp, get_ddp_rank, avg_ddp_metric
from .nn import from_pretrained
from .ops import Network
from torchvision.models.vision_transformer import _vision_transformer
try:
    from timm.optim import Lamb  # timm was not used in the paper's experiments, so it's optional
    from timm.loss import BinaryCrossEntropy
    from timm.data.mixup import Mixup
except Exception as e:
    print(e)

hf_token = "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"

login(token=hf_token, add_to_git_credential=True)  # need to login to download some models like Llama


log = partial(log, flush=True)
process = psutil.Process(os.getpid())



model_id = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
CONFIG = AutoConfig.from_pretrained(model_id)




class Trainer:
    def __init__(self,
                 model,
                 opt,
                 opt_args,
                 scheduler,
                 n_batches,
                 grad_clip=5,
                 auxiliary=False,
                 auxiliary_weight=0.4,
                 device='cuda',
                 log_interval=100,
                 predparam_wd=0,  # our predicted parameter regularization
                 scheduler_args=None,
                 save_dir=None,
                 ckpt=None,
                 epochs=None,
                 verbose=False,
                 amp=False,
                 amp_min_scale=None,            # 1024 for GHN-3
                 amp_growth_interval=2000,      # 100 for GHN-3
                 mixup=False,
                 compile_mode=None,
                 is_latent=False,
                 beta=1e-5,
                 ema=False,
                 ema_helper=None,
                 ):

        self.main_device = device  # where loss is computed
        # if bce:
        #     self.criterion = BinaryCrossEntropy(smoothing=label_smoothing).to(self.main_device)
        # else:
        #     self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(self.main_device)
        self.n_batches = n_batches
        self.grad_clip = grad_clip
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
        self.device = device
        self.log_interval = log_interval
        self.amp = amp
        self.amp_min_scale = amp_min_scale
        self.predparam_wd = predparam_wd
        self.epochs = epochs
        self.verbose = verbose
        self._is_latent = is_latent
        ### EMA ###
        self.ema = ema
        self.ema_helper = ema_helper
        ### EMA ###
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler(growth_interval=amp_growth_interval)
        self.ddp = is_ddp()
        if self.ddp:
            self.rank = get_ddp_rank()
            print("rank: ", self.rank)
            if self.verbose:
                print(f'trainer rank {self.rank}')
        else:
            self.rank = 0

        self.mixup_fn = Mixup(mixup_alpha=0.1, cutmix_alpha=1.0) if mixup else None
        if predparam_wd > 0:
            self.param_decay = lambda p: torch.norm(p, p='fro')

        model.to(self.rank if self.ddp else self.main_device)

        # Automatically resume from a checkpoint if exists or use GHN to initialize the model if ckpt is specified
        self.start_epoch = 0
        self.start_step = 0
        state_dict = None
        self.checkpoint_path = os.path.join(save_dir, 'checkpoint.pt') if save_dir else None
        if ckpt is not None or (self.checkpoint_path is not None and os.path.exists(self.checkpoint_path)):
            # Load model parameters from existing checkpoint
            if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
                # loads from save dir even if args.ckpt is specified
                ckpt = self.checkpoint_path
                log(f'Found existing checkpoint {ckpt} in the experiment directory {save_dir}.')

                log(f'Loading checkpoint from {ckpt}.')
                if self.ddp:
                    dist.barrier()  # make sure that all processes load the model before optimizing it
                    map_location = {'cuda:%d' % 0: device}
                else:
                    map_location = self.main_device
                state_dict = torch.load(ckpt, map_location=map_location)
                model.load_state_dict(state_dict['state_dict'])
                self.start_epoch = state_dict['epoch']
                self.start_step = state_dict['step']
                log('Model with {} parameters loaded from epoch {}, step {}.'.format(capacity(model)[1],
                                                                                     self.start_epoch,
                                                                                     self.start_step))
            else:
                ghn = from_pretrained(ckpt, debug_level=1).to('cpu')  # get a pretrained GHN
                model.to('cpu')
                model = ghn(model, bn_track_running_stats=True, keep_grads=False, reduce_graph=False)  # predict params
                model = init(model, orth=False, beta=beta)  # add a bit of noise to break symmetry of predicted params
                model.to(self.rank if self.ddp else self.main_device)

        self._is_ghn = isinstance(model, GHN) or (hasattr(model, 'module') and isinstance(model.module, GHN))
        if self.ddp:
            print("*******DDP RANK: ", self.rank)
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
            model._set_static_graph()

        if compile_mode not in [None, 'none', False]:
            try:
                log(f'compiling the model using the {compile_mode} mode to improve efficiency (if pytorch>=2.0)...')
                model = torch.compile(model, mode=compile_mode)
                log('compiling the model succeeded!')
            except Exception as e:
                log('compiling the model failed: %s' % e)

        self._model = model

        self._reset(opt, opt_args, scheduler, scheduler_args, state_dict)

    def reset_metrics(self, epoch):
        self._step = 0
        if epoch > self.start_epoch:
            self.start_step = 0
        self.metrics = {'loss': AvgrageMeter(), 'perplexity': AvgrageMeter()}
        if self._is_latent:
            self.metrics['gradnorm'] = AvgrageMeter()
        if self.predparam_wd > 0:
            self.metrics['loss_predwd'] = AvgrageMeter()  # predicted parameter regularization loss
        self.logger = Logger(self.n_batches, start_step=self.start_step)

    def _reset(self, opt, opt_args, scheduler, scheduler_args, state_dict):

        assert 'lr' in opt_args, 'learning rate must be specified in opt_args'
        if opt.lower() == 'sgd':
            optimizer = torch.optim.SGD
        else:
            if opt.lower() == 'adam':
                optimizer = torch.optim.Adam
            elif opt.lower() == 'adamw':
                optimizer = torch.optim.AdamW
            elif opt.lower() == 'lamb':
                optimizer = Lamb
            else:
                raise NotImplementedError(opt)
            if 'momentum' in opt_args:
                del opt_args['momentum']

        self._optimizer = optimizer(self._model.parameters(), **opt_args)

        if scheduler.startswith('cosine-warmup'):

            def parse_arg(arg, default):
                p = scheduler.find(arg)
                if p > 0:
                    p_end = scheduler[p:].find('-')
                    return float(scheduler[p + len(arg):len(scheduler) if p_end == -1 else p + p_end])
                else:
                    return default

            warmup_steps = int(parse_arg('steps', 5))  # number of warmup steps/epochs (default: 5)
            cycles = 0.5
            warmup_lr = parse_arg('init_lr', 1e-5) / opt_args['lr']  # initial warmup lr (default: 1e-5)

            def lr_lambda(step):
                # Based on https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
                if step < warmup_steps - 1:
                    return np.linspace(warmup_lr, 1, warmup_steps)[step]
                progress = float(step - warmup_steps) / float(max(1, self.epochs - warmup_steps))
                return max(0.0, 0.5 * (1. + math.cos(math.pi * cycles * 2.0 * progress)))
            self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda, verbose=self.verbose)

        elif scheduler == 'cosine':
            self._scheduler = CosineAnnealingLR(self._optimizer, self.epochs, verbose=self.verbose)
        elif scheduler == 'step':
            self._scheduler = StepLR(self._optimizer, verbose=self.verbose, **scheduler_args)
        elif scheduler == 'mstep':
            self._scheduler = MultiStepLR(self._optimizer, verbose=self.verbose, **scheduler_args)
            # e.g. GHN-2 scheduler_args={'milestones'=[200, 250], 'gamma'=0.1}
        else:
            raise NotImplementedError(scheduler)

        if state_dict is not None:
            if self.verbose:
                print('loading optimizer state')
            self._optimizer.load_state_dict(state_dict['optimizer'])

        # if training is resumed, adjust the learning rate
        if self.start_epoch > 0:
            self._scheduler.step(self.start_epoch)

        if self.amp:
            self.skipped_updates = 0

        self.reset_metrics(self.start_epoch)

        if state_dict is not None:
            if self.start_step >= self.n_batches - 1:
                self.start_step = 0
                self.start_epoch += 1  # resume from the next epoch
            else:
                self.start_step += 1  # resume from the next step

    def get_lr(self):
        for param_group in self._optimizer.param_groups:
            return param_group['lr']

    def scheduler_step(self):
        self._scheduler.step()

    def update(self, batch, models = None, graphs=None, embeddings=None, param_groups_map=None, ghn=None):





        def loss_check(loss_):
            if self.ddp:
                loss_avg_ = avg_ddp_metric(loss_)
                if torch.isnan(loss_avg_):
                    msg = f'rank {self.rank}, step {self._step}, the loss is {loss_}. ' \
                          f'Skip this batch, because the avg loss is {loss_avg_}.'
                    if self.verbose:
                        print(msg)
                    return msg
                else:
                    return loss_avg_
            elif torch.isnan(loss_):
                msg = f'the loss is {loss_}, unable to proceed. ' \
                      f'This issue may be fixed by restarting the script and loading the saved checkpoint ' \
                      f'using the --ckpt argument.'
                raise RuntimeError(msg)

            return loss_

        logits = []
        loss = 0
        loss_predwd = None
        nan_loss = torch.tensor(torch.nan, device=self.main_device)

        self._optimizer.zero_grad()
        if not self._model.training:
            self._model.train()

        try:
            with torch.cuda.amp.autocast(enabled=self.amp):

                if self._is_ghn:
                    # Predict parameters
                    if hasattr(graphs, 'nets') and len(graphs.nets) > 0:
                        models = graphs.nets
                    else:
                        # these are heavyweight Network objects that are less efficient but good for debugging
                        models = []
                        graphs2 = []
                        for nets_args in graphs.net_args:
                        #for config in graphs.configs:
                           
                            n_layer = np.random.randint(3, 10)
                            if n_layer > 7:
                                dim_min = 64
                                dim_max = 128
                            elif n_layer > 5:
                                dim_min = 128
                                dim_max = 192
                            else:
                                dim_min = 176
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
                            
                            intermediate_size_ratio = np.random.choice([3.5, 4])
                            n_key_value_heads = np.random.choice([1, 2, int(n_heads/2), n_heads])

                            # if n_layer > 5:
                            #     n_head = np.random.choice([2, 4])
                            # elif n_layer > 3:
                            #     n_head = np.random.choice([4, 6])
                            # else:
                            #     n_head = np.random.choice([6, 8])
                            
                            # if n_head == 8:
                            #     n_embd = np.random.choice(np.arange(72, 128 + 1, 8))
                            # elif n_head == 6:
                            #     n_embd = np.random.choice(np.arange(96, 192 + 1, 6))
                            # else:
                            #     n_embd = np.random.choice(np.arange(128, 384 + 1, 4))

                            
                            CONFIG.hidden_size = int(n_embd)
                            CONFIG.intermediate_size = int(n_embd * intermediate_size_ratio)
                            CONFIG.num_hidden_layers = int(n_layer)
                            CONFIG.num_attention_heads = int(n_heads)
                            CONFIG.num_key_value_heads = int(n_key_value_heads)
                            # print(CONFIG)

                            net = AutoModelForCausalLM.from_config(CONFIG)
                            # net = LlamaForCausalLM(CONFIG)
                            # net.config = config
                            # print(config)
                            # def get_var():
                            #     var = net(**tokenizer("Hello, my dog is cute", return_tensors="pt")).logits
                            #     return var
                            # net.get_var = get_var
                            
                            graph = Graph_LLM(net, tokenizer, ve_cutoff=250, dense=True, verbose=False)
                            
                            graphs2.append(graph)
                            models.append(net)

                    models = self._model(models,
                                         GraphBatch(graphs2, dense=True).to_device(self.main_device),    #to_device(self.device),
                                         bn_track_running_stats=True,
                                         keep_grads=True,
                                         reduce_graph=True
                                         )

                    if self.predparam_wd > 0:
                        total_norm = 0
                        for m in models:
                            for p in m.parameters():
                                total_norm += self.param_decay(p).to(self.main_device)

                        loss_predwd = self.predparam_wd * total_norm
                elif self._is_latent:
                    embeddings_new = self._model(embeddings)
                    models = ghn(models,
                                 GraphBatch(graphs2, dense=True).to_device(self.main_device),#to_device(self.device),
                                 embeddings=embeddings_new,
                                 param_groups_map=param_groups_map,
                                 bn_track_running_stats=True,
                                 keep_grads=True,
                                 )

                else:
                    models = self._model
            
                # model_device = models[0].device
                #print('model device: ', model_device)
                for key in batch:
                    batch[key] = batch[key].to(self.main_device, non_blocking=True)
               
                if not isinstance(models, (list, tuple)):
                    models = [models]

                for model in models:
                    try:
                        model = model.to(self.main_device)
                        outputs = model(**batch)
                    except:
                        print(model)
                        raise
                    
                    loss += outputs.loss
                    logits.append(outputs.logits.detach())
                    batch_size = outputs.logits.shape[0]
                
                logits = torch.stack(logits)

            if loss_predwd is not None:
                loss += loss_predwd

            loss = loss / len(logits)         # mean loss across models
            loss_avg = loss_check(loss)

            if self._step == 0 and self.ddp:
                if graphs is None:
                    net_idx = 0
                    n_graphs = 0
                else:
                    net_idx = graphs[0].net_idx
                    n_graphs = len(graphs)
                print(f'DDP: step {self._step}, rank {self.rank}, {n_graphs} graphs, '
                      f'net_idx {net_idx}, loss {loss}, loss_avg {loss_avg}, logits {logits.shape}')

            if isinstance(loss_avg, str):  # nan loss in any worker -> exit
                return loss_avg

            if self.amp:
                # Scales the loss, and calls backward()
                # to create scaled gradients
                self.scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self._optimizer)
            else:
                loss.backward()

            if self._step == 0 and self.rank == 0 and self.verbose:
                print_grads(self._model)

            if self.grad_clip > 0:
                parameters = []
                for group in self._optimizer.param_groups:
                    parameters.extend(group['params'])
                total_norm_clip = nn.utils.clip_grad_norm_(parameters, self.grad_clip)
            else:
                total_norm_clip = torch.zeros(1, device=self.main_device)

            if self.amp:
                # Unscales gradients and calls
                # or skips optimizer.step()
                retval = self.scaler.step(self._optimizer)

                if retval is None and torch.logical_or(total_norm_clip.isnan(), total_norm_clip.isinf()):
                    self.skipped_updates += 1

                # Updates the scale for next iteration
                self.scaler.update()

                if self.amp_min_scale is not None:
                    # if the scale is too small then training is hindered, so we manually keep the scale large enough
                    scale = self.scaler._check_scale_growth_tracker('update')[0]
                    if scale < self.amp_min_scale:
                        self.scaler._scale = torch.tensor(self.amp_min_scale).to(scale)
            else:
                self._optimizer.step()
            
            ### EMA ###
            if self.ema:
                self.ema_helper.update(self._model)
            ### EMA ###
            
            self.metrics['loss'].update((loss_avg if self.ddp else loss).item(), batch_size)
            self.metrics['perplexity'].update(math.exp((loss_avg if self.ddp else loss).item()), batch_size)

            if self._is_latent:
                self.metrics['gradnorm'].update((avg_ddp_metric(total_norm_clip)
                                                 if self.ddp else total_norm_clip).item(), batch_size)
            self._step += 1

        except RuntimeError as err:

            print('error', 'rank ', self.rank, type(err), err, graphs.net_args if graphs is not None else '')
            loss = nan_loss

            print(traceback.format_exc())
            print(traceback.print_exc())
            if not self.ddp:
                raise

        loss_avg = loss_check(loss)
        if isinstance(loss_avg, str):  # oom in any worker -> exit
            raise RuntimeError(loss_avg)

        return self.metrics

    def save(self, epoch, step, config, save_freq=300, interm_epoch=5):

        # save every save_freq steps, so that training can be resumed if one epoch takes a long time
        if not ((((step + 1) % save_freq == 0) or step == self.n_batches - 1) and self.rank == 0):
            return

        state_dict = {'state_dict': (self._model.module if hasattr(self._model, 'module')
                                     else self._model).state_dict(),
                      'optimizer': self._optimizer.state_dict(),
                      'epoch': epoch,
                      'step': step,
                      'ema_state_dict': self.ema_helper.state_dict() if self.ema else None,
                      }
        state_dict.update(config)
        torch.save(state_dict, self.checkpoint_path)
        log('\nsaved the checkpoint to {} at epoch={}, step={}'.format(self.checkpoint_path, epoch, step))

        if (epoch + 1) % interm_epoch == 0 or epoch == 0:
            checkpoint_path_interm = self.checkpoint_path.replace('.pt', '_epoch%d.pt' % (epoch + 1))
            torch.save(state_dict, checkpoint_path_interm)
            log('saved the intermediate checkpoint to {}'.format(checkpoint_path_interm))

    def log(self, step=None):
        step_ = self._step if step is None else (step + 1)
        if step_ % self.log_interval == 0 or step_ >= self.n_batches - 1 or step_ == 1:
            metrics = {metric: value.avg for metric, value in self.metrics.items()}
            if self.amp:
                metrics['amp_scale'] = self.scaler._check_scale_growth_tracker('update')[0].item()
            self.logger(step_, metrics)
