import os
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from collections import Counter

# Assuming you've adapted train_utils and fullflex_utils for PyTorch
from train_utils import AverageMeter, ce_loss, EMA, reduce_tensor, adjust_learning_rate
from .fullflex_utils import consistency_loss, cal_topK, nl_em_loss

import torch  # Import PyTorch instead of MegEngine
import torch.distributed as dist
import torch.nn.functional as F  # Import PyTorch functional

class FullFlex:
    def __init__(self, net_builder, num_classes, ema_m, p_cutoff, lambda_u, hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):

        super().__init__()
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.p_cutoff = p_cutoff  
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        self.optimizer = None
        self.ema = EMA(self.model, self.ema_m)
        self.it = 1
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, args, logger=None):
        labeled_epoch = 0
        unlabeled_epoch = 0
        fake_epoch = 0
        best_eval_acc, best_eval5_acc, best_it = 0.0, 0.0, 0

        save_path = os.path.join(args.save_dir, args.save_name)

        # PyTorch uses torch.ones and torch.zeros
        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.int32, device=args.device) * (-1) 
        classwise_acc = torch.zeros(args.num_classes, dtype=torch.float32, device=args.device)

        if args.resume:
            self.print_fn("==> Resuming from checkpoint..")
            assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
            # Load checkpoint using torch.load
            checkpoint = torch.load(args.resume, map_location='cpu')  
            best_eval_acc = checkpoint['best_acc']
            best_it = checkpoint['best_it']
            fake_epoch = checkpoint['epoch']
            labeled_epoch = checkpoint['labeled_epoch']
            unlabeled_epoch = checkpoint['unlabeled_epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            with open(os.path.join(save_path, 'selected_label_{}.pkl'.format(args.gpu)), 'rb') as f:
                selected_label = torch.tensor(pickle.load(f), device=args.device)
            dist.barrier()
            # Use torch.distributed for barrier synchronization
        
        if dist.get_world_size() > 1:
            # Broadcast model parameters and buffers
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)  
            for buffer in self.model.buffers():
                dist.broadcast(buffer.data, src=0)
            # Broadcast EMA model parameters and buffers (if using EMA)
            for param in self.ema.ema.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer.data, src=0)

        labeled_iter = iter(self.loader_dict['train_lb'])
        unlabeled_iter = iter(self.loader_dict['train_ulb'])

        losses = AverageMeter()
        losses_sup = AverageMeter()
        losses_unsup = AverageMeter()
        losses_npl = AverageMeter()
        losses_em = AverageMeter()
        mask_nums = AverageMeter()
        p_bar = tqdm(range(args.num_eval_iter), disable=(args.gpu!=0))
        start_iter = fake_epoch*self.num_eval_iter + 1


        for cur_iter in range(start_iter, args.num_train_iter+1):
            try:
                _, x_lb, y_lb = next(labeled_iter)
            except:
                labeled_iter = iter(self.loader_dict['train_lb'])
                _, x_lb, y_lb = next(labeled_iter)

            try:
                x_ulb_idx, x_ulb_w, x_ulb_s = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(self.loader_dict['train_ulb'])
                x_ulb_idx, x_ulb_w, x_ulb_s = next(unlabeled_iter)

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb = torch.tensor(x_lb, dtype=torch.float32, device=args.device)
            x_ulb_w = torch.tensor(x_ulb_w, dtype=torch.float32, device=args.device)
            x_ulb_s = torch.tensor(x_ulb_s, dtype=torch.float32, device=args.device)
            y_lb = torch.tensor(y_lb, dtype=torch.int32, device=args.device)

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset): 
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            inputs = F.concat((x_lb, x_ulb_w, x_ulb_s))
            
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:][:num_ulb], logits[num_lb:][num_ulb:]
            assert logits_x_ulb_w.shape[0] == logits_x_ulb_s.shape[0]
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                
            unsup_loss, mask, select, pseudo_lb, select_re = consistency_loss(logits_x_ulb_s, logits_x_ulb_w, classwise_acc, 'ce', self.p_cutoff, use_hard_labels=args.hard_label)

            if x_ulb_idx[select == 1].shape[0] != 0:
                selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

            k_value = cal_topK(logits_x_ulb_s.detach(), logits_x_ulb_w.detach(), topk=(2,args.num_classes))
            loss_npl, loss_em = nl_em_loss(logits_x_ulb_s, logits_x_ulb_w.detach(), k_value, select_re, self.p_cutoff)

            if args.loss_warm and fake_epoch < 5:
                total_loss = sup_loss + self.lambda_u * unsup_loss
            else:
                total_loss = sup_loss + self.lambda_u * unsup_loss + loss_npl + loss_em
                
            gm.backward(total_loss)
            self.optimizer.step().clear_grad()

            cur_lr = adjust_learning_rate(self.optimizer, cur_iter, args.num_train_iter, base_lr=args.lr)
            self.ema.update(self.model)

            if args.distributed:
                total_loss = reduce_tensor(total_loss.detach())
                sup_loss = reduce_tensor(sup_loss.detach())
                unsup_loss = reduce_tensor(unsup_loss.detach())
                loss_npl = reduce_tensor(loss_npl.detach())
                loss_em = reduce_tensor(loss_em.detach())
                mask = reduce_tensor(mask.detach())

            losses.update(total_loss.item())
            losses_sup.update(sup_loss.item())
            losses_unsup.update(unsup_loss.item())
            losses_npl.update(loss_npl.item())
            losses_em.update(loss_em.item())
            mask_nums.update(mask.item())

            p_bar.set_description("Train Epoch: {epoch}. Iter: {batch:4}. LR: {lr:.4f}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_n: {loss_n:.4f}. Loss_e: {loss_e:.4f}. Mask: {mask_nums:.4f}. K: {k_value: d}".format(
                    epoch=fake_epoch,
                    batch=cur_iter-self.num_eval_iter*fake_epoch,
                    lr=cur_lr,
                    loss=losses.avg,
                    loss_x=losses_sup.avg,
                    loss_u=losses_unsup.avg,
                    loss_n=losses_npl.avg,
                    loss_e=losses_em.avg,
                    mask_nums=mask_nums.avg,
                    k_value=k_value))
            p_bar.update()
                
            if cur_iter % self.num_eval_iter == 0:
                p_bar.close()
                tb_dict = {}
                top1, top5 = self.evaluate(self.ema.ema, args=args)
                if args.gpu==0:
                    tb_dict['eval/1.top-1-acc'] = top1
                    tb_dict['eval/2.top-5-acc'] = top5

                    tb_dict['train/1.losses'] = losses.avg
                    tb_dict['train/2.losses_sup'] = losses_sup.avg
                    tb_dict['train/3.losses_unsup'] = losses_unsup.avg
                    tb_dict['train/4.losses_npl'] = losses_npl.avg
                    tb_dict['train/4.losses_em'] = losses_em.avg
                    tb_dict['train/5.mask'] = mask_nums.avg

                    if tb_dict['eval/1.top-1-acc'] > best_eval_acc:
                        best_eval_acc = tb_dict['eval/1.top-1-acc']
                        best_eval5_acc = tb_dict['eval/2.top-5-acc']
                        best_it = self.it

                    self.print_fn('EVAL/Epoch_{}: TOP1: {}, TOP5: {}'.format(fake_epoch, top1, top5))
                    self.print_fn(f"BEST_EVAL_ACC: {best_eval_acc}, BEST_EVAL_TOP5_ACC: {best_eval5_acc}")

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, fake_epoch)

                    self.save_model({
                            'acc': top1,
                            'best_acc': best_eval_acc,
                            'best_it': best_it,
                            'epoch': fake_epoch + 1,
                            'labeled_epoch': labeled_epoch,
                            'unlabeled_epoch': unlabeled_epoch,
                            'state_dict': self.model.state_dict(),
                            'ema_state_dict': self.ema.ema.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, (self.it==best_it), save_path)
                
                pickle.dump(selected_label.tolist(), open(os.path.join(save_path, 'selected_label_{}.pkl'.format(args.gpu)), 'wb'))
                dist.group_barrier()
                losses.reset()
                losses_sup.reset()
                losses_unsup.reset()
                losses_npl.reset()
                losses_em.reset()
                mask_nums.reset()
                p_bar = tqdm(range(args.num_eval_iter), disable=(args.gpu!=0))
                fake_epoch += 1
                
            self.it += 1
            # print(time.perf_counter()-start)

        if not self.tb_log is None:
            self.tb_log.close()
    def evaluate(self, test_model, eval_loader=None, args=None):
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
    
        top1 = AverageMeter()
        top5 = AverageMeter()
    
        test_model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculations for evaluation
            for (inputs, targets) in eval_loader:  
              inputs = inputs.to(args.device)  # Move inputs to the correct device
              targets = targets.to(args.device) # Move targets to the correct device
            
            outputs = test_model(inputs)  
            
            # Calculate accuracy using torchmetrics if available, otherwise use custom function
            try:
                import torchmetrics
                acc1 = torchmetrics.functional.accuracy(outputs, targets, top_k=1)  
                acc5 = torchmetrics.functional.accuracy(outputs, targets, top_k=5) 
            except ImportError:
                # Custom accuracy calculation (adapt from your previous code)
                _, pred = outputs.topk(5, 1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))

                acc1 = correct[:1].reshape(-1).float().sum(0) * 100.0 / targets.size(0)  
                acc5 = correct[:5].reshape(-1).float().sum(0) * 100.0 / targets.size(0)
           
            # Reduce accuracy across processes if using distributed training
            if dist.is_initialized() and dist.get_world_size() > 1:
                acc1 = reduce_tensor(acc1, world_size=dist.get_world_size())  
                acc5 = reduce_tensor(acc5, world_size=dist.get_world_size()) 
                
            top1.update(acc1.item(), inputs.size(0))  
            top5.update(acc5.item(), inputs.size(0))
            
        return top1.avg, top5.avg

        
