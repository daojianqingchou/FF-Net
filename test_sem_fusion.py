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
    spatial_shape = conf.spatial_shape
    
    if isinstance(spatial_shape, int):
        spatial_shape = [spatial_shape] * 3
        
    classifier = get_model(model, num_class=num_class, point_dims=conf.point_dims, voxel_dims=conf.voxel_dims, spatial_shape=spatial_shape)
    try:
        log.info(f'loading from {conf.model_path}')
        checkpoint = torch.load(conf.model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log.info(f'Use pretrain model {classifier}')
    except:
        log.info('No existing model, starting training from scratch...')
        start_epoch = 0
        

    # dataset    
    data_path = conf.data_path
    test_dataset = get_dataset('plant_dataset_voxel', data_path, 'test', spatial_shape=spatial_shape)
    
    batch_size = conf.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=PlantVoxelDataset.collate_fn)
    
    global_epoch = 0
    best_iou = 0

    classifier.cuda()
        
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
        log.info(f"model {model} trained on {conf.model_path} are evaluated on {data_path[data_path.rindex(os.sep)+1:data_path.rindex('.')]}")
        log.info(f'eval point accuracy: {total_correct / total_seen}')
        log.info(f'eval point class accuracy: {total_correct_class / total_seen_class}')
        log.info(f'eval class IoU (ground, leaf, stem): {total_correct_class / (total_union_class + 1e-6)}')
        log.info(f'eval avg class IoU: {mIoU}')        
            
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-conf', default='configs/test_ff_conf.yaml')

    args = parser.parse_args()
    
    main(args)

