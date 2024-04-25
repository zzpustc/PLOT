#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random

from copy import deepcopy

from configs.template import config
from datasets.Samplers import ClassAwareSampler
from datasets.ClassPrioritySampler import ClassPrioritySampler

from utils.lr_scheduler import adjust_learning_rate
from utils.loss import mixup_criterion
from utils.pytorch import grad_norm
from utils.pareto import pareto_search, epo_search

from utils.utils import (
    create_logger, 
    Averager, 
    shot_acc, 
    deep_update_dict,
    get_optimizer,
    get_scheduler,
    pre_compute_class_ratio,
    freeze_backbone,
    mixup_data,
    lr_reset
)

def train_sample(epoch, train_loader, model, optimizer, logger, class_weights):

    model.train()
    
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    ta = Averager()

    for step, (x, y, _) in enumerate(train_loader):
        
        x, y = x.cuda(), y.cuda()

        _, fea, o = model(x)
        fea.requires_grad = True

        o = model.module.classifier(fea)
        loss_ori = F.cross_entropy(o, y, reduction = 'none')

        if cfg['train']['sampler'] == 'IS':
            loss = loss_ori
            
        elif cfg['train']['sampler'] == 'Decoup':

            y_in = y.detach().cpu().numpy()
            loss_list = []
            alpha = (epoch - cfg['train']['stage'] + 1) * 1.0 / (cfg['train']['max_epoch'] - cfg['train']['stage'])
            alpha = alpha if alpha < cfg['train']['up_limit'] else cfg['train']['up_limit']
            for y_tmp in np.unique(y_in): 
                idx = np.where(y_in == y_tmp)   
                loss_cls_spc = loss_ori[idx]
                if len(idx[0]) > 1:
                    loss_cls_spc = loss_ori[idx].mean()
                
                fea_grad = torch.autograd.grad(loss_cls_spc, fea, retain_graph = True, allow_unused=True)[0]
                fea_grad = fea_grad / torch.norm(fea_grad)
                noise = cfg['adver_reg'] * fea_grad * class_weights[y_tmp] 
                fea_new = fea + noise 
                o_tmp = model.module.classifier(fea_new)

                loss_tmp = F.cross_entropy(o_tmp, y, reduction = 'none') * class_weights[y_tmp] 
                loss_list.extend(loss_tmp[idx])

            loss_list = torch.stack(loss_list)
            loss = alpha * loss_list.mean() +  (1 - alpha) * loss_ori.mean()
            loss_flat = loss_list.mean().item()

        pred_q = F.softmax(o, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item() 
        acc = correct * 1.0 / y.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        tl.add(loss.item()) 
        ta.add(acc)

        if step % cfg['print_inteval'] == 0:
            print(('Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f}').format(train_loss = loss.item(), ori_loss = loss_ori.mean().item(), flat_loss = loss_flat, train_acc = acc))
            logger.info(('Trainnig Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Flat Loss:{flat_loss:.3f}, Training Acc:{train_acc:.2f}').format(train_loss = loss.item(), ori_loss = loss_ori.mean().item(), flat_loss = loss_flat, train_acc = acc))
            

    loss_ave = tl.item()
    acc_ave = ta.item()
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Average Loss:{loss_ave:.3f}, Average Acc:{acc_ave:.2f}').format(epoch=epoch, loss_ave=loss_ave, acc_ave = acc_ave))
    
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Acc:{acc_ave:.2f}').format(epoch=epoch, loss_ave=loss_ave, acc_ave = acc_ave))

    return model


def train(epoch, train_loader, model, optimizer, logger, class_ratio, class_weights, grad_dims):


    
    model.train()

    # ----- RECORD LOSS AND ACC ----- 
    tl = Averager()
    ta = Averager()
    
    for step, (x, y, _) in enumerate(train_loader):
        
        x, y = x.cuda(), y.cuda()

        if cfg['train']['mixup']:
            criterion = nn.CrossEntropyLoss(reduction = 'none').cuda()
            images, targets_a, targets_b, lam = mixup_data(x, y, cfg['train']['mixup_alpha'])
            fea, _, o = model(images)
            loss_ori = mixup_criterion(criterion, o, targets_a, targets_b, lam)
        else:
            fea, _, o = model(x)
            loss_ori = F.cross_entropy(o, y, reduction = 'none')


        loss =  loss_ori.mean()

        pred_q = F.softmax(o, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item() 
        acc = correct * 1.0 / y.shape[0]

        optimizer.zero_grad()

        # specified for each classes
        if epoch < cfg['train']['pareto_start_epoch']: 
            loss.backward(retain_graph=True)
            y_in = y.detach().cpu().numpy()
 
            pareto_module = model.module.layer3   
            pareto_module_params = list(pareto_module.parameters())   
            device = pareto_module_params[0].device
            # calculate gradient norm here
            grad_c_norm = grad_norm(pareto_module_params, device) + 1e-12

            # Apply SAM here
            origin_params = deepcopy(pareto_module_params)
            for param in pareto_module_params:
                grad_c = param.grad
                denominator = grad_c / grad_c_norm
                noise = cfg['train']['rho'] * 1.0 * denominator
                param.data = param.data + noise

            output = model.module.forward_again(fea)
            loss_sam = F.cross_entropy(output, y, reduction = 'none')

            var_cons_sam = 0
            count_class = len(np.unique(y_in))
            for y_tmp in np.unique(y_in):
                idx = np.where(y_in == y_tmp)
                if len(idx[0]) > 1:
                    loss_c = loss_sam[idx]
                    var_c = torch.std(loss_c)
                    var_cons_sam += var_c
                else:
                    count_class -= 1
            var_cons_sam = var_cons_sam / count_class
            loss_sam_mean = loss_sam.mean() + cfg['train']['bal_ratio'] * var_cons_sam
            
            optimizer.zero_grad()
            loss_sam_mean.backward(retain_graph=True)

            grads = torch.Tensor(sum(grad_dims), len(np.unique(y_in))).cuda()
            cnt = 0
            for y_tmp in np.unique(y_in):
                grads[:, cnt].fill_(0.0)
                idx = np.where(y_in == y_tmp)
                f_param_grads = torch.autograd.grad(loss_sam[idx].sum() / cfg['train']['batch_size'], pareto_module_params, retain_graph=True) 
                for ii in range(len(grad_dims)):
                    beg = 0 if ii == 0 else sum(grad_dims[:ii])
                    en = sum(grad_dims[:(ii + 1)])
                    grads[beg:en, cnt].copy_(f_param_grads[ii].data.view(-1))
                cnt += 1
                del f_param_grads

            pareto_search(epoch, cfg, pareto_module, grads, grad_dims, class_weights, np.unique(y_in))
            for ii in range(len(pareto_module_params)):
                pareto_module_params[ii].data = cfg['train']['perturb_radius'] * pareto_module_params[ii].data + (1 - cfg['train']['perturb_radius']) * origin_params[ii].data.clone()

        else:
            loss.backward()

        optimizer.step()

        tl.add(loss.item()) 
        ta.add(acc) 

        if step % cfg['print_inteval'] == 0:
            print(('Training Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Training Acc:{train_acc:.2f}').format(train_loss = loss.item(), ori_loss = loss_ori.mean().item(), train_acc = acc))
            logger.info(('Training Loss:{train_loss:.3f}, Ori Loss:{ori_loss:.3f}, Training Acc:{train_acc:.2f}').format(train_loss = loss.item(), ori_loss = loss_ori.mean().item(), train_acc = acc))
            

    loss_ave = tl.item()
    acc_ave = ta.item()
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Average Loss:{loss_ave:.3f}, Average Acc:{acc_ave:.2f}').format(epoch=epoch, loss_ave=loss_ave, acc_ave = acc_ave))
    
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Acc:{acc_ave:.2f}').format(epoch=epoch, loss_ave=loss_ave, acc_ave = acc_ave))
    
    return model


def val(epoch, val_loader, model, logger, train_dataset):
    
    model.eval()
    # ----- RECORD LOSS AND ACC ----- 
    total_logits = torch.empty((0, cfg['setting']['num_class'])).cuda().float()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    
    with torch.no_grad():
        for step, (x, y, _) in enumerate(val_loader):
        
            x, y = x.cuda(), y.cuda()

            _, _, o = model(x)
            loss = F.cross_entropy(o, y, reduction = 'none')
            loss = loss.mean()
        
            pred_q = F.softmax(o, dim=1)
            total_logits = torch.cat((total_logits, pred_q))
            total_labels = torch.cat((total_labels, y))
            
            pred_q = pred_q.argmax(dim=1)
            correct = torch.eq(pred_q, y).sum().item() 
            acc = correct * 1.0 / y.shape[0]
        
            if step % cfg['print_inteval'] == 0:
                print(('Testing Loss:{val_loss:.3f},  Testing Acc:{val_acc:.2f}').format(val_loss = loss.item(), val_acc = acc))
                logger.info(('Testing Loss:{val_loss:.3f},  Testing Acc:{val_acc:.2f}').format(val_loss = loss.item(), val_acc = acc))
    
    total_logits = total_logits.argmax(dim=1)
    
    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1, overall_acc = shot_acc(total_logits, total_labels, train_dataset)
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Many:{many_acc:.4f},  Medium:{median_acc:.4f}, Low:{low_acc:.4f}, Overall:{overall_acc:.4f}').format(epoch=epoch, many_acc=many_acc_top1, median_acc = median_acc_top1, low_acc = low_acc_top1, overall_acc = overall_acc))
    
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Many:{many_acc:.4f},  Medium:{median_acc:.4f}, Low:{low_acc:.4f}, Overall:{overall_acc:.4f}').format(epoch=epoch, many_acc=many_acc_top1, median_acc = median_acc_top1, low_acc = low_acc_top1, overall_acc = overall_acc))
    return many_acc_top1, median_acc_top1, low_acc_top1, overall_acc

if __name__ == '__main__':

	# ----- LOAD PARAM -----
    # path = "./configs/Cifar10_200.json" 
    path = "./configs/Cifar100_50.json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=path)

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  

    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)
    
    feature_param = {'use_modulatedatt': False, 'use_fc': False, 'dropout': None,
                 'stage1_weights': False, 'caffe': True}
    
    # ----- SET DATALOADER -----
    if cfg['dataset']['dataset_name'] == 'Cifar100':
        from datasets.Cifar import IMBALANCECIFAR100
        from models.ResNet32Feature import create_model

        LT_dataset = cfg['dataset']['dataset_name'] + '_LT'
        train_dataset = IMBALANCECIFAR100(phase = 'train', imbalance_ratio=cfg['train']['cifar_imb_ratio'], root=cfg['dataset']['data_root'])
        val_dataset = IMBALANCECIFAR100(phase = 'val', imbalance_ratio= None, root=cfg['dataset']['data_root'], reverse = 0)
        test_dataset = IMBALANCECIFAR100(phase = 'test', imbalance_ratio= None, root=cfg['dataset']['data_root'], reverse = 0)

    elif cfg['dataset']['dataset_name'] == 'Cifar10':
        from datasets.Cifar import IMBALANCECIFAR10
        from models.ResNet32Feature import create_model

        LT_dataset = cfg['dataset']['dataset_name'] + '_LT'
        train_dataset = IMBALANCECIFAR10(phase = 'train', imbalance_ratio=cfg['train']['cifar_imb_ratio'], root=cfg['dataset']['data_root'])
        val_dataset = IMBALANCECIFAR10(phase = 'val', imbalance_ratio= None, root=cfg['dataset']['data_root'], reverse = 0)
        test_dataset = IMBALANCECIFAR10(phase = 'test', imbalance_ratio= None, root=cfg['dataset']['data_root'], reverse = 0)


    if cfg['train']['sampler'] == 'IS':
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], pin_memory=True)
    elif cfg['train']['sampler'] == 'CS':
        casampler = ClassAwareSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset, sampler = casampler, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'], pin_memory=True)
    elif cfg['train']['sampler'] == 'PBS':
        casampler = ClassPrioritySampler(train_dataset, epochs = cfg['train']['max_epoch'])
        train_loader = DataLoader(dataset=train_dataset, sampler = casampler, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'], pin_memory=True)
    else:
        IS_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], pin_memory=True, drop_last = True)
        casampler = ClassAwareSampler(train_dataset)
        CS_loader = DataLoader(dataset=train_dataset, sampler = casampler, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'], pin_memory=True)


    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg['test']['batch_size'], shuffle=True, num_workers=cfg['test']['num_workers'], pin_memory=True)    
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=cfg['test']['num_workers'], pin_memory=True)   

    # PRE-DEFINE CLASS WEIGHTS
    _, class_ratio, class_weights = pre_compute_class_ratio(cfg, train_dataset)

    # ----- MODEL -----
    model = create_model(cfg, *feature_param).cuda()
    model = nn.DataParallel(model)

    # for model module selection
    grad_dims = []
    pareto_module_params = list(model.module.layer3.parameters())  # should be modified to layer_3
    for param in pareto_module_params:
        grad_dims.append(param.data.numel())

    # ----- OPTIMIZER -----
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, cfg['train']['stage'])

    best_h_acc = 0
    best_acc = 0
    
    for epoch in range(cfg['train']['max_epoch']):
        print(('Epoch {epoch:d} is pending...'
                 ).format(epoch = epoch))
        logger.info(('Epoch {epoch:d} is pending...'
                 ).format(epoch = epoch))

        # ----- FOR STAGE-1 -----
        if epoch < cfg['train']['stage']:
            scheduler.step()
            train_loader = IS_loader
            model = train(epoch, train_loader, model, optimizer, logger, class_ratio, class_weights, grad_dims)

        # ----- FOR STAGE-2 -----
        else:
            if epoch == cfg['train']['stage']:
                optimizer = lr_reset(cfg, model)   # RESET LR
                weights_name = cfg['save_dir'] + str(cfg['train']['cifar_imb_ratio']) + str(cfg['train']['rho']) + '_' + str(cfg['train']['bal_ratio']) + '_' + str(cfg['train']['pareto_start_epoch'])  + '_best_model.pth'
                state_dict = torch.load(weights_name)
                model.load_state_dict(state_dict)

            train_loader = CS_loader 
            freeze_backbone(model)

            adjust_learning_rate(optimizer, epoch - cfg['train']['stage'], cfg)
            model = train_sample(epoch, train_loader, model, optimizer, logger, class_weights)
        
        # ----- TESTING -----
        h_acc, m_acc, t_acc, acc = val(epoch, val_loader, model, logger, train_dataset)

        if epoch % cfg['save_fre'] == 0 and epoch > 0:
            weights_name = cfg['save_dir'] + cfg['backbone']['name'] + '_model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), weights_name)
         
        if acc > best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            logger.info('Find a better model and save it!')       

            weights_name = cfg['save_dir'] + str(cfg['train']['cifar_imb_ratio']) + str(cfg['train']['rho']) + '_' + str(cfg['train']['bal_ratio']) + '_' + str(cfg['train']['pareto_start_epoch'])  + '_best_model.pth'
            torch.save(model.state_dict(), weights_name)

        print('Best Accuracy is: ', best_acc)
        
