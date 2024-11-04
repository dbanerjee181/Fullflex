import os
import logging
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
# torch.autograd (Autograd is built-in, no need for a separate import)
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging

from train_utils import TBLog, get_optimizer  # Assuming these are adapted for PyTorch
from utils import get_logger, net_builder, str2bool, over_write_args_from_file  # Assuming these are adapted for PyTorch
from models.fullflex.fullflex import FullFlex  # Assuming this is adapted for PyTorch
from datasets.ssl_dataset import SSL_Dataset  # Assuming this is adapted for PyTorch

def worker(args):
#    dist.init_process_group(
#        backend='nccl',  # Or 'gloo' if using CPUs
#        init_method='env://',  # Or your preferred init method
#        world_size=args.world_size,
#        rank=args.ngpus  # Assuming args.gpu is the rank
#    )

#    args.world_size = dist.get_world_size()
#    args.gpu = dist.get_rank()  # In PyTorch, this is usually called 'rank' or 'local_rank'
    args.gpu = 1
    save_path = os.path.join(args.save_dir, args.save_name)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)  # Use torch.manual_seed for PyTorch
        torch.cuda.manual_seed(args.seed) # Use torch.cuda.manual_seed for PyTorch
        np.random.seed(args.seed)

    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    
    if args.gpu == 0:
        tb_log = TBLog(args.save_dir, args.save_name)  # Assuming TBLog is adapted for PyTorch
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)  # Assuming get_logger is adapted for PyTorch
    logger.warning(f"USE GPU: {args.gpu} for training")  

    if args.dataset.upper() == 'CIFAR100' and args.num_labels == 400 and args.world_size > 1:
        args.sync_bn = True  # You might need to handle SyncBatchNorm differently in PyTorch

    args.bn_momentum = 0.999
  
    _net_builder = net_builder(args.net, args.net_from_name,
                                    {'first_stride': 2 if 'stl' in args.dataset else 1,
                                    'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout,
                                    'use_embed': False,
                                    'is_remix': False,
                                    'sync_bn': args.sync_bn},)

    model = FullFlex(_net_builder, args.num_classes, args.ema_m, args.p_cutoff, args.ulb_loss_ratio, args.hard_label,
                        num_eval_iter=args.num_eval_iter, tb_log=tb_log, logger=logger)

    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    model.set_optimizer(optimizer)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint file found!"
        checkpoint = torch.load(args.resume, map_location='cpu')  # Load checkpoint using torch.load
        
        # Load model state dictionary
        model.model.load_state_dict(checkpoint['state_dict'])  
        
        # Load EMA state dictionary (if using EMA)
        if 'ema_state_dict' in checkpoint:
            model.ema.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        # Barrier synchronization (if using distributed training)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    args.batch_size = int(args.batch_size / args.world_size)
    logger.info(f"model_arch: {model}")

    if args.dataset != "imagenet":
        # Barrier synchronization (if using distributed training and not on rank 0)
        if dist.is_initialized() and dist.get_world_size() > 1 and args.gpu != 0:
            dist.barrier()
        
        # Create SSL_Dataset instance (adapted for PyTorch)
        train_dset = SSL_Dataset(args, name=args.dataset, train=True, num_classes=args.num_classes, data_dir=args.data_dir)
        
        # Get labeled and unlabeled datasets
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)  # Assuming get_ssl_dset is adapted for PyTorch

        # Create evaluation dataset
        _eval_dset = SSL_Dataset(args, name=args.dataset, train=False, num_classes=args.num_classes, data_dir=args.data_dir)
        eval_dset = _eval_dset.get_dset()  # Assuming get_dset is adapted for PyTorch
        
        # Barrier synchronization (if using distributed training and on rank 0)
        if dist.is_initialized() and dist.get_world_size() > 1 and args.gpu == 0:
            dist.barrier()
    else:
        print('Please Waiting for Supporting')
        exit()
    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    # Create data loaders using torch.utils.data.DataLoader
    loader_dict['train_lb'] = data.DataLoader(
        dset_dict['train_lb'],
        sampler=data.RandomSampler(dset_dict['train_lb']),  # Use RandomSampler for training
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True  # Drop the last incomplete batch if necessary
    )
    loader_dict['train_ulb'] = data.DataLoader(
        dset_dict['train_ulb'],
        sampler=data.RandomSampler(dset_dict['train_ulb']),  # Use RandomSampler for training
        batch_size=args.batch_size * args.uratio,
        num_workers=args.num_workers,
        drop_last=True  # Drop the last incomplete batch if necessary
    )
    loader_dict['eval'] = data.DataLoader(
        dset_dict['eval'],
        batch_size=args.eval_batch_size,  # Use batch_size directly for evaluation
        shuffle=False,  # Do not shuffle for evaluation
        num_workers=4,
        drop_last=False  # Do not drop the last incomplete batch if necessary
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()

    trainer = model.train
    trainer(args, logger=logger)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FullMatch Training')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fullflex')
    parser.add_argument('--resume', type=str,default=None)

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2**20, help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=1024, help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7, help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='batch size of evaluation data loader')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--loss_warm', type=bool, default=False)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')

    '''
    multi-GPUs & Distrbitued Training
    '''
    parser.add_argument('-n','--ngpus', default=8, type=int,help='number of GPUs per node (default: None, use all available GPUs)',)
    parser.add_argument('--dist-addr', default='localhost')
    parser.add_argument('--dist-port', default=23456, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)


    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    args.cur_dir = os.getcwd().split('/')[-1]

    args.distributed = False

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork('gpu')

    args.world_size = 1  
    worker(args)
    