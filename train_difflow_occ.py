import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging
import importlib

from tqdm import tqdm 
# from models_bid_lighttoken_res import PointConvBidirection
# from models_bid_lighttoken_new import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *


def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    if args.multi_gpu is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/PointConv%diters_%s-'%(args.iters,args.dataset) + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.model_name + '.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    # os.system('cp %s %s' % ('train_bid_iter_cosine.py', log_dir))
    os.system('cp %s %s' % ('config_train_occ.yaml', log_dir))
    #os.system('cp %s %s' % ('train_difflow_occ.py', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    module = importlib.import_module(args.model_name)
    # model = PointConvBidirection()
    model = getattr(module, 'PointConvBidirection')(iters=args.iters)
    multiScaleLoss = getattr(module, 'multiScaleLoss')

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            args.num_points),
        num_points=args.num_points,
        data_root = args.data_root,
        full=args.full
    )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        # worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    val_dataset = datasets.__dict__[args.val_dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.val_path
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    if args.pretrain is not None:
        save_model = torch.load(args.pretrain)
        model_dict = model.state_dict()
        #del_list = ['recurrent0.flow.fc.weight','recurrent0.flow.fc.bias','recurrent1.flow.fc.weight',\
                    #'recurrent1.flow.fc.bias','recurrent2.flow.fc.weight','recurrent2.flow.fc.bias',\
                      #'flow3.fc.weight', 'flow3.fc.bias' ]
        state_dict = {key: value for key, value in save_model.items() \
                      if key in model_dict and model_dict[key].shape == value.shape}
        model.load_state_dict(state_dict, strict = False)
        #model.load_state_dict(torch.load(args.pretrain),strict=False)
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    if torch.cuda.is_available():
        # 获取可用的CUDA设备数量
        num_devices = torch.cuda.device_count()
        print("Obtain the number of available CUDA devices:", num_devices)
        if num_devices > 1:
            torch.backends.cudnn.benchmark = True

            model = torch.nn.DataParallel(model)
            model.cuda()
        else:
            model.cuda()
    else:
        raise EnvironmentError("CUDA is not available.")

    pretrain = args.pretrain 
    #init_epoch = 0 
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0 

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
                      
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5 

    history = defaultdict(lambda: list())
    uncertainty = args.uncertainty
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        #eval_epe3d, eval_loss = eval_sceneflow(model.eval(), val_loader, multiScaleLoss)
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if (epoch+1)%10 == 0:
            uncertainty = max(0.02, uncertainty*0.8)
        print('uncertainty uncertainty:%f'%uncertainty)

        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9, ncols=100,mininterval=60.0):
            pos1, pos2, norm1, norm2, flow, mask = data  
            #move to cuda 
            pos1 = pos1.cuda().float()
            pos2 = pos2.cuda().float()
            norm1 = norm1.cuda().float()
            norm2 = norm2.cuda().float()
            flow = flow.cuda().float()
            mask = mask.unsqueeze(2).cuda().float()

            model = model.train() 
            pred_flows, fps_pc1_idxs, _, _, _ ,loss_f2,loss_f1,loss_f0, _mask, _ = model(pos1, pos2, norm1, norm2, flow, mask, uncertainty = uncertainty)

            loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs,_mask,loss_f2,loss_f1,loss_f0)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step() 
            optimizer.zero_grad()

            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        print(str_out)
        logger.info(str_out)

        eval_epe3d, eval_loss = eval_sceneflow(model.eval(), val_loader, multiScaleLoss)
        str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_loss)
        print(str_out)
        logger.info(str_out)

        if eval_epe3d < best_epe:
            best_epe = eval_epe3d
            if args.multi_gpu is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best epe loss is: %.5f'%(best_epe))
        logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader, multiScaleLoss):

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9, mininterval=10.0):
        pos1, pos2, norm1, norm2, flow, mask = data  
        
        #move to cuda 
        pos1 = pos1.cuda().float()
        pos2 = pos2.cuda().float()
        norm1 = norm1.cuda().float()
        norm2 = norm2.cuda().float()
        flow = flow.cuda().float()
        mask = mask.unsqueeze(2).cuda().float()

        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, _, _, _mask, gt_flows= model(pos1, pos2, norm1, norm2, flow, mask)

            eval_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs, _mask)

            mask_sum = torch.sum(_mask[0], 1)  # B 1
            error = torch.norm(
                pred_flows[0][0].permute(0, 2, 1) - gt_flows[0],
                dim=2,
                keepdim=True,
            )  # B 2048
            epe3d = (
                torch.sum(error * _mask[0], 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
            )
            epe3d = torch.mean(epe3d)

        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_eval = np.mean(metrics['eval_loss'])

    return mean_epe3d, mean_eval

if __name__ == '__main__':
    main()




