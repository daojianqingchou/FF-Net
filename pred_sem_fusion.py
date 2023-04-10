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
from plant_dataset_voxel import voxelize
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
from sample_util import sample_random
import h5py

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
    criterion = get_loss(model, weight=torch.tensor(conf.weight))
    try:
        log.info(f'loading from {conf.model_path}')
        checkpoint = torch.load(conf.model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log.info(f'Use pretrain model {classifier}')
    except Exception as e:
        print(e)
        log.info('No existing model...')
        
    # dataset    
    # s_point = conf.s_point
    data_path = conf.data_path
    data_names = conf.data_names
    
    for data_name in data_names:
        # data_name = conf.data_name
        
        ground_path = os.path.join(data_path[:data_path.rindex(os.sep)], '_'.join([data_name, 'ground.txt']))
        pred_path = os.path.join(data_path[:data_path.rindex(os.sep)], '_'.join([data_name, 'pred.txt']))
        
        log.info(f'ground_path: {ground_path}')
        log.info(f'pred_path: {pred_path}')
        log.info(f'data_name: {data_name}')
        
        with h5py.File(data_path) as h5_data:
            data = h5_data['test'][data_name]
            data, label = sample_random(data[:, :3], data[:, 3], 2048, align=False)
    
    
        classifier.cuda()     
        # evaluate
        with torch.no_grad():
            classifier = classifier.eval()
            
            
            ground_points = []
            pred_points_list = []
                
                
            for i, (pc, label) in tqdm(enumerate(zip(data, label)), total=len(data)):

                _, voxel_label, pc, pc_indices, label = voxelize(pc, label, spatial_shape)
                
                # target_voxel = [torch.from_numpy(v).to(conf.device) for v in voxel_label]
                # target_voxel = torch.stack(target_voxel, dim=0)
                # pc = [torch.from_numpy(p).to(conf.device) for p in pc]
                # label = [torch.from_numpy(l) for l in label]
                # pc_indices = [torch.from_numpy(p).to(conf.device) for p in pc_indices]
                pc = [torch.from_numpy(pc).to(conf.device, torch.float)]
                pc_indices = [torch.from_numpy(pc_indices).to(conf.device, torch.int32)]
    
                pred_voxel, pred_points = classifier(pc, pc_indices)
                pred_voxel = torch.softmax(pred_voxel, dim=1)
                
                # target_point = torch.cat(label, dim=0).to(torch.long)
                # target_point = np.concatenate(label, axis=0)
                target_point = [label[:, None]]
                
                
                
                pred_voxel_label = torch.argmax(pred_voxel, dim=1).cpu().numpy()
    
                pc_indices = [pc_ind.cpu().numpy() for pc_ind in pc_indices]
                pred_point = []
                # print(pred_voxel_label.shape)
                for i, pc_ind in enumerate(pc_indices):
                    pred_point.append(pred_voxel_label[i, pc_ind[:, 0], pc_ind[:, 1], pc_ind[:, 2]])
                pred_point = np.concatenate(pred_point, axis=0)
                
                # pc = [p.cpu().detach().squeeze() for p in pc]
                pc = pc[0]
                # label = 

                ground_points.append(np.concatenate([pc[:, :3].cpu().detach().squeeze(), label[:, None]], axis= -1))
                pred_points_list.append(np.concatenate([pc[:, :3].cpu().detach().squeeze(), pred_point.squeeze()[:, None]], axis = -1))
            
        ground_points = np.concatenate(ground_points, axis=0)
        pred_points = np.concatenate(pred_points_list, axis=0)   
    
        np.savetxt(ground_path, ground_points)
        np.savetxt(pred_path, pred_points)
    log.info('prediction finished...')
           
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-conf', default='configs/pred_conf.yaml')

    args = parser.parse_args()
    
    main(args)

