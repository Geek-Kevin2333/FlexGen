#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  Train GPT-2 Medium with LoRA (see our paper for hyperparameters for GPT-2 Medium)
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import torch
import random
from torch.utils.data import DataLoader
torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    """ todo what is the _optimizer, what is clip?
    _loss: the loss function to optimize
    _optimizer: the optimizer to use for optimization (e.g. SGD or Adam)
    _model: the model being optimized
    _schedule: an optional learning rate scheduler
    args: a collection of arguments used for optimization (e.g. learning rate, batch size, etc.)
    is_update: a boolean flag that specifies whether to update the model or perform validation only

    The function first computes the gradients of the loss function using backpropagation. If _schedule is not None, it steps the scheduler by one epoch.
    If is_update is True, it then performs the optimization step by calling _optimizer.step(), which updates the parameters of the model using the computed gradients.
    If args.clip is greater than 0, it applies gradient clipping to prevent exploding gradients.
    Finally, it zeros out the gradients of _optimizer using _optimizer.zero_grad() to prepare for the next iteration of optimization.
    If is_update is False, the function does not update the model but only computes the loss function without backpropagation.
    """
    # todo 如何做到部分更新的，即只更新lora矩阵。是不是设置requiresGradient的值为false/true即可
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    if is_update:
        if args.clip > 0:
            if args.fp16:
                # The norm is computed over all gradients together, as if they were concatenated into a single vector.
                # Gradients are modified in-place.
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()        
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}
            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    args, 
    train_step=0, 
    epoch=0
):
    """
    model: the language model being trained
    optimizer: the optimizer to be used for training (e.g. SGD or Adam)
    scheduler: the learning rate scheduler to be used for training
    train_loader: the data loader for the training data
    valid_loader: the data loader for the validation data
    args: a collection of arguments used for training (e.g. learning rate, batch size, etc.)
    train_step: the starting step for training (default is 0)
    epoch: the starting epoch for training (default is 0)
    """
    # The function first sets the model in training mode using model.train().
    # It also initializes the AverageMeter() class to keep track of the average loss during training.
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    train_loader.sampler.set_epoch(epoch)

    # For each batch of data in the train_loader, the function performs the forward pass through the model using the input (_input), target (_target), and mask (_msk) data.
    # _lm_logits contains the predicted logits while _lm_loss contains the loss that is computed using the predicted logits and the true targets.
    # The function calls the optimizer_step() function to update the model with the computed loss and to perform optimization.
    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)


        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        ) 

        _lm_loss = _lm_loss.mean() 

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )
        # The function also checks whether the current step is a multiple of args.log_interval, args.save_interval, or args.eval_interval.
        # If it is, it performs the appropriate operation such as printing the current training status, saving a checkpoint, or evaluating the current performance on the validation data.

        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if args.rank == 0: 
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()
        
        if train_step % args.save_interval == 0: 
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print('saving checkpoint', model_path)
                torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)
            distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)

            model.train()
            distributed_sync(args)

        if train_step == args.max_step:
            break
    # Finally, the function checks whether the maximum number of steps (args.max_step) has been reached and saves the final model to disk.
    # It then returns the final training step (train_step).
    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path) 
    distributed_sync(args)
    return train_step


if __name__ == '__main__':
    # args example
    # python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    #     --train_data ./data/e2e/train.jsonl \
    #     --valid_data ./data/e2e/valid.jsonl \
    #     --train_batch_size 8 \
    #     --grad_acc 1 \
    #     --valid_batch_size 4 \
    #     --seq_len 512 \
    #     --model_card gpt2.md \
    #     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    #     --platform local \
    #     --clip 0.0 \
    #     --lr 0.0002 \
    #     --weight_decay 0.01 \
    #     --correct_bias \
    #     --adam_beta2 0.999 \
    #     --scheduler linear \
    #     --warmup_step 500 \
    #     --max_epoch 5 \
    #     --save_interval 1000 \
    #     --lora_dim 4 \
    #     --lora_alpha 32 \
    #     --lora_dropout 0.1 \
    #     --label_smooth 0.1 \
    #     --work_dir ./trained_models/GPT2_M/e2e \
    #     --random_seed 110
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)
    # 从英伟达网页Apex (A PyTorch Extension) — Apex 0.1.0 documentation可以得到apex的全称——A PyTorch Extension(Apex)
    # ——其实就是一种pytorch的拓展插件。它的目的就是为了是用户能够快速实现amp——自动混合精度技术在它们的N卡系列上训练模型也构建的一个库。
    # 简简单单的在你自定义的模型训练代码中添加少量代码(4行代码)就能够使用自动混合精度的技术来提高模型的训练速度，提高生产力。简单的理解apex就是一个用来支持模型训练在pytorch框架下使用混合精度进行加速训练的拓展插件之类的库；也可以理解为一种模型训练加速技术——其实是amp自动混合精度搭配硬件一起才能加速的。
    # amp——auto mixed  precision——自动半精度，它的最核心的东西在于低精度Fp16。它能够提供一种非常可靠和友好的方式进行模型在Fp16精度下进行训练。

    # # Initialization of command-line arguments, random number generators, and data loaders:
    # # The script starts by parsing command-line arguments using the argparse library. It then initializes the random number generators for both Python and PyTorch (torch.manual_seed and random.seed). Next, the script creates the training and validation data loaders using the FT_Dataset class and the DataLoader class from PyTorch. The FT_Dataset class initializes the text data from a specified file, tokenizes it, and converts it into a Tensor that is used as input for the language model. The DataLoader class creates a batch sampler and worker threads to load batches of data.
    # # Creation of the GPT2 language model:
    # # The script initializes the GPT2 language model based on the configuration specified in the command-line arguments. The GPT2LMModel class is used to create the model, which is then loaded with pre-trained weights if args.init_checkpoint is specified. The lora.mark_only_lora_as_trainable(lm_net) function is used to mark the LoRA part of the model as trainable, so that only the attention weights for LoRA are updated during optimization.
    # #
    # # Setup of optimizer and optimizer scheduler:
    # # The create_adam_optimizer_from_args function is used to create the Adam optimizer based on the configuration specified in the command-line arguments. The create_optimizer_scheduler function is used to create the optimizer scheduler based on the optimizer and the configuration arguments.
    # #
    # # Training and validation of the language model:
    # # The train_validate function is used to perform training and validation of the language model for the specified number of epochs. This function first initializes the AverageMeter() class to keep track of the average loss during training. For each batch of data, the train_validate function performs the forward pass through the model using the input, target, and mask data. The _lm_logits tensor contains the predicted logits while _lm_loss contains the loss that is computed using the predicted logits and the true targets. The function calls the optimizer_step() function to update the model with the computed loss and to perform optimization.
    # #
    # # Synchronization and cleanup:
    # # After training is complete, the script synchronizes the processes and cleans up the distributed environment using the distributed_sync and cleanup functions.
    # #
    # # Overall, LoRA operations are implemented in the GPT2LMModel class, which is used to cr

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )     
    
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint))    
    #  Moves all model parameters and buffers to the GPU.
    #  This also makes associated parameters and buffers different objects. \
    # todo maybe we can evaluate the different offloading policies  tensor placement、approximate method、computation delegation
    # lmModel是Model的lora版本，需要自己构造optimizer、optimizer_scheduler。和原model共享model_embeddings_weights
    lm_net = lm_net.cuda()

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    # todo distributed_opt？
    lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                train_step=train_step, epoch=epoch
            )
            
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')

    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)

