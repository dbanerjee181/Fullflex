import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.distributed as dist

from copy import deepcopy
import os 
import math

def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    # PyTorch optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    # Add other optimizers if needed (e.g., Adam)
    elif optim_name == 'Adam':
        optimizer = optim.Adam(per_param_args, lr=lr, weight_decay=weight_decay)
    
    return optimizer


def adjust_learning_rate(optimizer, current_step, num_training_steps, num_cycles=7. / 16., num_warmup_steps=0, base_lr=0.03):
    """Adjusts the learning rate using a cosine annealing schedule with optional warmup.

    Args:
        optimizer: The PyTorch optimizer.
        current_step: The current training step.
        num_training_steps: The total number of training steps.
        num_cycles: The number of cosine cycles (default: 7/16).
        num_warmup_steps: The number of warmup steps (default: 0).
        base_lr: The base learning rate (default: 0.03).

    Returns:
        The adjusted learning rate.
    """
    if current_step < num_warmup_steps:
        _lr = float(current_step) / float(max(1, num_warmup_steps))
    else:
        num_cos_steps = float(current_step - num_warmup_steps)
        num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
        _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
    _lr = _lr * base_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = _lr
    return _lr

class EMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay

    def update(self, model):
        state_dict = model.state_dict()
        ema_state_dict = self.ema.state_dict()
        
        for name, param in state_dict.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            ema_state_dict[name].data.copy_(
                ema_state_dict[name].data * self.decay +
                param.data * (1. - self.decay)
            )
            
        self.ema.load_state_dict(ema_state_dict)

class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None, mode="train"):
        if suffix is None:
            suffix = ''
        for key, value in tb_dict.items():
            self.writer.add_scalar(suffix + key, value, it)
        self.writer.flush()
            
    def close(self):
        self.writer.close()


class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
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


def ce_loss(logits, targets, use_hard_labels=True, reduction='mean'):
    """
    Calculates the cross-entropy loss.

    Args:
        logits: The predicted logits.
        targets: The ground truth labels.
        use_hard_labels: Whether to use hard labels or soft labels (not used in this implementation).
        reduction: The reduction method to apply to the loss.

    Returns:
        The calculated cross-entropy loss.
    """
    log_pred = F.log_softmax(logits, dim=-1)  # Change 'axis' to 'dim'
    # Gather operation is replaced by indexing
    loss = -log_pred.gather(dim=1, index=targets.unsqueeze(1)).squeeze() 
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()


def reduce_tensor(tensor, mean=True):
    """
    Reduces a tensor across all processes in a distributed setting.

    Args:
        tensor: The tensor to reduce.
        mean: If True, calculates the mean across processes. 
              If False, calculates the sum across processes.

    Returns:
        The reduced tensor.
    """
    rt = tensor.clone()  # Create a copy to avoid modifying the original tensor
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # Sum across processes
    if mean:
        rt /= dist.get_world_size()  # Calculate the mean
    return rt