from statistics import variance
import torch
import torch.nn.functional as F

config=dict(
    dataset=dict(
        dataset_name='', # choose from ['Places', 'ImageNet', 'iNat2018', 'CIFAR']
        data_root='',
        open_root = ''
    ),
    head=dict(
        type='MLP', # choose from  ['FCNorm', 'LWS', 'MLP']  
        hidden_dims_lst=[256, 256, 256],
        bias=True,
    ),
    backbone=dict(
        name ='ResNet-152', # choose from ['ResNet-50', 'ResNet-101']
        freeze=False, 
        pretrain=False,
        hidden_dim = 512
    ),
    loss = dict(
        type = 'CrossEntropy'
    ),
    train=dict(
        batch_size=128,
        div = 1,
        disturb = 'backbone', # ['backbone', 'head', 'hybrid']
        max_epoch=50,
        distributed=False,
        stage = 30,
        random_times = 1,
        local_rank=0,
        num_workers=16,
        shuffle=True,
        sampler= 'IS', # choose from ['IS', 'CS', 'PBS', 'Decoup']
        optimizer=dict(
            type='ADAM', # choose from ['SGD', 'ADAM']
            lr=1e-3, # learning rate
            lr_cls = 1e-1,
            lr_hyper = 1e-1, # lr of hyper param
            momentum=0.9,
            wc=2e-4, # weight decay
        ),
        lr_scheduler=dict(
            type='multistep', # choose from ['multistep', 'cosine', 'warmup']
            lr_step= [20],
            lr_factor=0.1,
            warmup_epoch=20,
            cosine_decay_end=0,
        ),
        two_stage=dict(
            drw=False,
            drs=False,
            start_epoch=1,
        ),
        tensorboard=dict(
            enable=True,
        )
    ),
    test=dict(
        batch_size=64,
        exp_id='',
        resume_head = '',
        num_workers=8,
        error = 1.0e-1,
        lamda = 1000,
    ),
    test_open = dict(
        flag = True,
        batch_size = 64,
        model_file = 'best_model.pth',
        num_workers=8,
        shot = 5,
        open_threshold = 0.1,
        num_dim = 4, # select the subspace dim
        resume = ''
    ),
    setting=dict(
        type='LT Classification', # choose from ['Imbalanced Learning', 'LT Classification', 'LT Regression', 'Open LT']
        num_class= 365, # only effective for LT Regression
    ),
    eval_mode=False,
    output_dir='./output/logs/',
    save_dir = './output/weights/',
    seed=2,  # [0, 1, 42]
    use_gpu=True,
    gpu_id= 0,
    resume_model='',
    resume_mode='all',
    valid_step=5,
    pin_memory=True,
    save_fre = 10,
    print_inteval = 20,
    debug = True,
    variance = 1e-4
)
