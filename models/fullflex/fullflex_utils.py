import torch
import torch.nn.functional as F  # Import PyTorch's functional module
import numpy as np

from train_utils import ce_loss, reduce_tensor

def consistency_loss(logits_s, logits_w, class_acc, name='ce', p_cutoff=0.95, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    assert name == 'ce', 'must ce'

    # PyTorch uses softmax and argmax from torch.nn.functional
    pseudo_label = F.softmax(logits_w, dim=-1)  
    max_probs, max_idx = torch.max(pseudo_label, dim=-1) 

    # Use torch.where for conditional masking
    mask = torch.where(max_probs >= p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx])), 
                       torch.tensor(1.0, device=logits_s.device), 
                       torch.tensor(0.0, device=logits_s.device)).type(torch.float32)
    select = torch.where(max_probs >= p_cutoff, 
                        torch.tensor(1, device=logits_s.device), 
                        torch.tensor(0, device=logits_s.device)).type(torch.int32)
    
    if use_hard_labels:
        # Assuming ce_loss is adapted for PyTorch
        masked_loss = ce_loss(logits_s, max_idx.long(), use_hard_labels, reduction='none') * mask 
    else:
        print('must use hard label')

    return masked_loss.mean(), mask.mean(), select, max_idx, mask.type(torch.int32)

def nl_em_loss(pred_s, pred_w, k, mask_pred, p_cutoff):
    # Softmax calculations
    softmax_pred = F.softmax(pred_s, dim=-1)  # Use dim=-1 in PyTorch
    pseudo_label = F.softmax(pred_w, dim=-1)  # Use dim=-1 in PyTorch
    
    # Top-k indices
    _, topk = torch.topk(pseudo_label, k, dim=-1)  # Get indices in PyTorch
    
    # Create mask_k (one-hot encoding of top-k indices)
    mask_k = torch.zeros_like(pseudo_label, device=pred_s.device)  
    mask_k.scatter_(1, topk, 1)  # Use scatter_ for in-place assignment
    
    # Create mask_k_npl (masking for non-pseudo-labeled samples)
    mask_k_npl = torch.where((mask_k == 1) & (softmax_pred > p_cutoff**2), 
                            torch.tensor(0.0, device=pred_s.device), 
                            mask_k)
    
    # Calculate loss_npl
    loss_npl = (-torch.log(1 - softmax_pred + 1e-10) * mask_k_npl).sum(dim=1).mean()  # Use dim=1 in PyTorch
    
    # Get pseudo labels (argmax of pseudo_label)
    label = torch.argmax(pseudo_label, dim=-1)  
    
    # Update mask_k to include pseudo-labeled samples
    mask_k.scatter_(1, label.unsqueeze(1), 1)  # Use unsqueeze to add dimension
    
    # Calculate yg (sum of softmax_pred for pseudo-labeled samples)
    yg = softmax_pred.masked_select(mask_k.bool()).reshape(pred_w.shape[0], -1).sum(dim=-1, keepdim=True)  
    
    # Calculate soft_ml (soft pseudo-label weight)
    soft_ml = (1 - yg + 1e-7) / (k - 1)
    soft_ml = soft_ml.expand_as(pred_s)  # Use expand_as in PyTorch
    
    # Create mask for EM loss
    mask = 1 - mask_k
    mask = mask * mask_pred.unsqueeze(1)  # Use unsqueeze to add dimension
    mask = torch.where((mask == 1) & (softmax_pred > p_cutoff**2), 
                       torch.tensor(0.0, device=pred_s.device), 
                       mask)
    
    # Calculate loss_em (EM loss)
    loss_em = -(soft_ml * torch.log(softmax_pred + 1e-10) + 
                 (1 - soft_ml) * torch.log(1 - softmax_pred + 1e-10))
    
    # ... (Combine loss_npl and loss_em as needed)
    loss_em = (loss_em * mask).sum()/(mask.sum()+1e-10)
    return loss_npl, loss_em

def cal_topK(pred_s, pred_w, topk=(1,)):
    # Get target labels (argmax of pred_w)
    target_w = torch.argmax(pred_w, dim=-1)  # Use dim=-1 in PyTorch

    output = pred_s
    target = target_w
    
    maxk = max(topk)
    batch_size = target.size(0)  # Get batch size using size(0) in PyTorch

    # Get top-k predictions
    _, pred = output.topk(maxk, dim=1)  # Use dim=1 for topk along dimension 1
    
    # Check correctness
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Use eq() for element-wise equality
    
    # Calculate accuracy for different k values
    for k in range(topk[0], topk[1] + 1):
        correct_k = correct[:, :k].reshape(-1).float().sum(0)  # Sum correct predictions for top-k
        acc_single = correct_k * (100.0 / batch_size)  # Calculate accuracy
        
        # Reduce accuracy across processes if using distributed training
        if dist.is_initialized() and dist.get_world_size() > 1:
            acc_parallel = reduce_tensor(acc_single, world_size=dist.get_world_size())  
        else:
            acc_parallel = acc_single

        if acc_parallel > 99.99:
            return k  # Return k if accuracy is above threshold