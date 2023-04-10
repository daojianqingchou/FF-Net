'''
Created on 2023年2月23日

@author: dongzi
'''

import os
import time
from model import get_model, get_loss
from dataset import get_dataset
# from dataset.plant_dataset import PlantDataset
from plant_dataset_voxel import PlantVoxelDataset
import logging
from utils.log_util import LogUtil
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict
import torch
import torch.nn.functional as F
from utils import provider
import numpy as np
import einops as eo


def main(args):
    # config
    conf = args.conf
    with open(conf) as stream:
        conf = yaml.safe_load(stream)
        conf = EasyDict(conf)
        
    # log
    log_dir = conf.log_dir
    log_dir = os.path.join(conf.log_dir, time.strftime('%Y%m%d'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = LogUtil(os.path.join(log_dir, time.strftime('%H%M%S')+'.log'))
    
    # model
    model = conf.model
    num_class = conf.num_class
    
    classifier = get_model(model, spatial_shape=conf.spatial_shape, num_class=conf.num_class, point_dims=conf.point_dims, voxel_dims=conf.voxel_dims)
    criterion = get_loss(model, weight=1/torch.tensor(conf.weight))
    criterion = criterion.to(conf.device)
    
    check_path = os.path.join(log_dir, 'checkpoints', time.strftime('%H%M%S'))
    if not os.path.exists(check_path):
        os.makedirs(check_path)
        
    try:
        checkpoint = torch.load(os.path.join(check_path, 'best_model.pth'))
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log.info('No existing model, starting training from scratch...')
        start_epoch = 0

    # optimizer
    if conf.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=conf.learning_rate, momentum=0.9)

    # dataset    
    data_path = conf.data_path
    train_dataset = get_dataset('plant_dataset_voxel', data_path, 'train', spatial_shape=conf.spatial_shape)
    test_dataset = get_dataset('plant_dataset_voxel', data_path, 'test', spatial_shape=conf.spatial_shape)

    batch_size = conf.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PlantVoxelDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=PlantVoxelDataset.collate_fn)
    
    global_epoch = 0
    best_iou = 0

    classifier.cuda()
    # training 
    for epoch in range(start_epoch, conf.epoch):
        classifier.train()
        
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        
        for i, (_, voxel_label, pc, pc_indices, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            target_voxel = [torch.from_numpy(v).to(conf.device) for v in voxel_label]
            target_voxel = torch.stack(target_voxel, dim=0)
            pc = [torch.from_numpy(p).to(conf.device) for p in pc]
            label = [torch.from_numpy(l).to(conf.device) for l in label]
            pc_indices = [torch.from_numpy(p).to(conf.device) for p in pc_indices]
            

            pred_voxel, pred_points = classifier(pc, pc_indices)
            pred_voxel = torch.softmax(pred_voxel, dim=1)
            
            target_point = torch.cat(label, dim=0).to(torch.long)
            
            
            pred_voxel_label = torch.softmax(pred_voxel, dim=1)
            pc_indices = [pc_ind.to(torch.long) for pc_ind in pc_indices]
            pred_point = []
            # print(pred_voxel_label.shape)
            for i, pc_ind in enumerate(pc_indices):
                # print(pc_ind.shape)
                pred_point.append(pred_voxel_label[i,:, pc_ind[:, 0], pc_ind[:, 1], pc_ind[:, 2]].T)
            pred_point = torch.cat(pred_point).to(conf.device)
            # print(pred_point.shape)
            
            # target_voxel = target_voxel.to(torch.long)
            loss = criterion(pred_point, target_point, pred_voxel, target_voxel)
            # loss = criterion(pred_voxel, target_voxel, None, None)
            
            loss.backward()
            optimizer.step()
            
            seg_pred_choice = np.argmax(pred_point.cpu().detach().numpy(), -1)
            target_point = target_point.cpu().numpy()
            total_correct += np.sum(target_point == seg_pred_choice)
            
            # print(target_point.shape, seg_pred_choice.shape)
            # print(total_correct)
            total_seen += len(target_point)
            loss_sum += loss
            
        log.info(f'Training mean loss: {loss_sum / num_batches}')
        log.info(f'Training accuracy: {total_correct / float(total_seen)}')
        
        # evaluate
        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            # labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = np.zeros(num_class)
            total_correct_class = np.zeros(num_class)
            total_union_class = np.zeros(num_class)
            classifier = classifier.eval()
            
            log.info(f'evaluation for epoch {epoch} ...')
            
            for i, (_, voxel_label, pc, pc_indices, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
                
                # target_voxel = [torch.from_numpy(v).to(conf.device) for v in voxel_label]
                # target_voxel = torch.stack(target_voxel, dim=0)
                pc = [torch.from_numpy(p).to(conf.device) for p in pc]
                # label = [torch.from_numpy(l) for l in label]
                pc_indices = [torch.from_numpy(p).to(conf.device) for p in pc_indices]
                
    
                pred_voxel, pred_points = classifier(pc, pc_indices)
                pred_voxel = torch.softmax(pred_voxel, dim=1)
                
                # target_point = torch.cat(label, dim=0).to(torch.long)
                target_point = np.concatenate(label, axis=0)
                
                
                pred_voxel_label = torch.argmax(pred_voxel, dim=1).cpu().numpy()

                pc_indices = [pc_ind.cpu().numpy() for pc_ind in pc_indices]
                pred_point = []
                # print(pred_voxel_label.shape)
                for i, pc_ind in enumerate(pc_indices):
                    pred_point.append(pred_voxel_label[i, pc_ind[:, 0], pc_ind[:, 1], pc_ind[:, 2]])
                pred_point = np.concatenate(pred_point, axis=0)
                
                
                total_correct += np.sum(pred_point == target_point)
                total_seen += len(target_point)

                # print(pred_point.shape, target_point.shape)                
                for i in range(num_class):
                    total_seen_class[i] += np.sum(target_point == i)
                    total_correct_class[i] += np.sum((pred_point == i) & (target_point == i))
                    total_union_class[i] += np.sum((pred_point == i) | (target_point == i))
            
            mIoU = np.mean(total_correct_class / (total_union_class + 1e-6))
            log.info(f'eval point accuracy: {total_correct / total_seen}')
            log.info(f'eval point class accuracy: {total_correct_class / total_seen_class}')
            log.info(f'eval class IoU (ground, leaf, stem): {total_correct_class / (total_union_class + 1e-6)}')
            log.info(f'eval avg class IoU: {mIoU}')        
            
            
            if mIoU > best_iou:
                state_dir = os.path.join(log_dir, 'models')
                if not os.path.exists(state_dir):
                    os.makedirs(state_dir)
                    
                # model_path = os.path.join(state_dir, 'best_model.pth')
                model_path = os.path.join(state_dir, '_'.join([time.strftime('%H%M%S'), 'best_model.pth']))
                
                best_iou = mIoU
                log.info('saving model ...')
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, model_path)
                
            log.info(f'best mIoU: {best_iou}')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-conf', default='configs/maize_conf.yaml')

    args = parser.parse_args()
    
    main(args)

