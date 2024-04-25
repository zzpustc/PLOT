#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from copy import deepcopy

import numpy as np
import argparse
import json
import random

from configs.template import config
from utils.pytorch import set_gpu_mode
from utils.utils import (
    deep_update_dict,
    create_logger,
    Averager,
    get_optimizer,
    get_scheduler,
    shot_acc,
    transform_selection,
    pre_compute_class_ratio,
    reset_weight
	)
from datasets.Places365 import LT_Dataset
# from models.ResNet152Feature import create_model
from models.ResNet50Feature import create_model

from models.DotProductClassifier import create_cls

def test_ensemble(cfg, val_loader, test_loader, model, logger, train_dataset):

    head_model = deepcopy(model)
    weights_name = cfg['save_dir']  + 'head_expert.pth'
    head_model.load_state_dict(torch.load(weights_name))

    # calibration here
    head_model = val_scaling(cfg, val_loader, head_model, expert = 'head')
 #   tail_model = val_scaling(cfg, val_loader, model, expert = 'tail')
    tail_model = model
    
    tail_model.eval()
    head_model.eval()

    best_alpha = 0.2

    # record loss and acc
    total_logits = torch.empty((0, cfg['setting']['num_class'])).cuda().float()
    total_logits_head = torch.empty((0, cfg['setting']['num_class'])).cuda().float()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    
    with torch.no_grad():
        for step, (x, y, _) in enumerate(test_loader):
        
            x, y = x.cuda(), y.cuda()
            _, _, o = tail_model(x)
            loss = F.cross_entropy(o, y)
        
            pred_q = F.softmax(o, dim=1) #.argmax(dim=1)
            total_logits = torch.cat((total_logits, pred_q))
            total_labels = torch.cat((total_labels, y))

            o = head_model(x)
            pred_q = F.softmax(o, dim=1)
            total_logits_head = torch.cat((total_logits_head, pred_q))
            
            pred_q = pred_q.argmax(dim=1)
            correct = torch.eq(pred_q, y).sum().item() 
            acc = correct * 1.0 / y.shape[0]
        
            if step % cfg['print_inteval'] == 0:
                print(('Testing Loss:{val_loss:.3f},  Testing Acc:{val_acc:.2f}').format(val_loss = loss.item(), val_acc = acc))
                logger.info(('Testing Loss:{val_loss:.3f},  Testing Acc:{val_acc:.2f}').format(val_loss = loss.item(), val_acc = acc))

    total_logits += best_alpha * total_logits_head
    total_logits = total_logits.argmax(dim=1)

    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1, overall_acc = shot_acc(total_logits, total_labels, train_dataset)
    
    print('++++++++++++++++++++++++++Testing+++++++++++++++++++++++++++++')
    print(('Many:{many_acc:.4f},  Median:{median_acc:.4f}, Low:{low_acc:.4f}, Overall:{overall_acc:.4f}').format(many_acc=many_acc_top1, median_acc = median_acc_top1, low_acc = low_acc_top1, overall_acc = overall_acc))
    print('\n')
    
    logger.info('++++++++++++++++++++++++++Testing+++++++++++++++++++++++++++++')
    logger.info(('Many:{many_acc:.4f},  Median:{median_acc:.4f}, Low:{low_acc:.4f}, Overall:{overall_acc:.4f}').format(many_acc=many_acc_top1, median_acc = median_acc_top1, low_acc = low_acc_top1, overall_acc = overall_acc))
    logger.info('\n')

    return overall_acc

def test_api(cfg, test_loader, model, logger, train_dataset):
    
    model.eval()
    
    # record loss and acc
    CE = nn.CrossEntropyLoss()
    total_logits = torch.empty((0, cfg['setting']['num_class'])).cuda().float()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    
    with torch.no_grad():
        for step, (x, y, _) in enumerate(test_loader):
        
            x, y = x.cuda(), y.cuda()
            _, _, o = model(x)
            loss = F.cross_entropy(o, y)
        
            pred_q = F.softmax(o, dim=1) #.argmax(dim=1)
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
    
    print('++++++++++++++++++++++++++Testing+++++++++++++++++++++++++++++')
    print(('Many:{many_acc:.4f},  Median:{median_acc:.4f}, Low:{low_acc:.4f}, Overall:{overall_acc:.4f}').format(many_acc=many_acc_top1, median_acc = median_acc_top1, low_acc = low_acc_top1, overall_acc = overall_acc))
    print('\n')
    
    logger.info('++++++++++++++++++++++++++Testing+++++++++++++++++++++++++++++')
    logger.info(('Many:{many_acc:.4f},  Median:{median_acc:.4f}, Low:{low_acc:.4f}, Overall:{overall_acc:.4f}').format(many_acc=many_acc_top1, median_acc = median_acc_top1, low_acc = low_acc_top1, overall_acc = overall_acc))
    logger.info('\n')
    
    return overall_acc   


if __name__ == '__main__':
    
	# ----- Load Param -----
    # path = "./configs/ImageNet_LT.json"
    path = "./configs/Cifar10_200.json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=path)

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)  # update part of params in config   

    # set the fix sampled tasks
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    
    # define model
    LT_dataset = cfg['dataset']['dataset_name'] + '_LT'
    dataset_root = cfg['dataset']['data_root'] + '/' + cfg['dataset']['dataset_name'] 
    # ----- SET GPU ID -----
    device = set_gpu_mode(cfg['use_gpu'], cfg['gpu_id'])

    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank, test=True)

    # define dataset
    feature_param = {'use_modulatedatt': False, 'use_fc': False, 'dropout': None,
                 'stage1_weights': False, 'caffe': True}
    # ----- SET DATALOADER -----
    if cfg['dataset']['dataset_name'] == 'Places':
        from datasets.Places365 import LT_Dataset
        from models.ResNet152Feature import create_model

        LT_dataset = cfg['dataset']['dataset_name'] + '_LT'
        dataset_root = cfg['dataset']['data_root'] + '/' + cfg['dataset']['dataset_name'] 
            
        train_txt = cfg['dataset']['data_root'] + LT_dataset + '/' + LT_dataset + '_train.txt'
        train_dataset  = LT_Dataset(dataset_root, train_txt, transform_selection(cfg, 'train'))

        val_txt = cfg['dataset']['data_root']  + LT_dataset + '/' + LT_dataset + '_val.txt'
        val_dataset  = LT_Dataset(dataset_root, val_txt, transform_selection(cfg, 'val'))

        test_txt = cfg['dataset']['data_root'] + LT_dataset + '/' + LT_dataset + '_test.txt'
        test_dataset  = LT_Dataset(dataset_root, test_txt, transform_selection(cfg, 'test'))

        feature_param['dataset'] = LT_dataset

    elif cfg['dataset']['dataset_name'] == 'ImageNet':
        from datasets.Places365 import LT_Dataset
        from models.ResNet50Series import create_model

        LT_dataset = cfg['dataset']['dataset_name'] + '_LT'
        dataset_root = cfg['dataset']['data_root'] + '/' + cfg['dataset']['dataset_name'] 
            
        train_txt = cfg['dataset']['data_root'] + LT_dataset + '/' + LT_dataset + '_train.txt'
        train_dataset  = LT_Dataset(dataset_root, train_txt, transform_selection(cfg, 'train'))

        val_txt = cfg['dataset']['data_root']  + LT_dataset + '/' + LT_dataset + '_val.txt'
        val_dataset  = LT_Dataset(dataset_root, val_txt, transform_selection(cfg, 'val'))

        test_txt = cfg['dataset']['data_root'] + LT_dataset + '/' + LT_dataset + '_test.txt'
        test_dataset  = LT_Dataset(dataset_root, test_txt, transform_selection(cfg, 'test'))

    elif cfg['dataset']['dataset_name'] == 'iNat2018':
        from datasets.Places365 import LT_Dataset
        from models.ResNet50Series import create_model

        dataset_root = cfg['dataset']['data_root']
            
        train_txt = cfg['dataset']['data_root'] + 'iNaturalist18_train.txt'
        train_dataset  = LT_Dataset(dataset_root, train_txt, transform_selection(cfg, 'train'))

        val_txt = cfg['dataset']['data_root'] + 'iNaturalist18_val.txt'
        val_dataset  = LT_Dataset(dataset_root, val_txt, transform_selection(cfg, 'val'))

        test_txt = cfg['dataset']['data_root'] + 'iNaturalist18_val.txt'
        test_dataset  = LT_Dataset(dataset_root, test_txt, transform_selection(cfg, 'test'))


    elif cfg['dataset']['dataset_name'] == 'Cifar100':
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


    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=cfg['test']['num_workers'], pin_memory=True)    
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=cfg['test']['num_workers'], pin_memory=True)     

    _, class_ratio, class_weights, _ = pre_compute_class_ratio(cfg, train_dataset)
    class_weights = torch.from_numpy(class_weights)
    feature_param = {'use_modulatedatt': False, 'use_fc': True, 'dropout': None,
                 'stage1_weights': False, 'dataset': LT_dataset, 'caffe': True}
    
    model = create_model(cfg, *feature_param).cuda()
    model = nn.DataParallel(model).cuda()

    # resume weights
    model = reset_weight(model, cfg['test_open']['resume'])
    acc = test_api(cfg, test_loader, model, logger, train_dataset)
    print('Test Acc: ', acc)
    
    
    