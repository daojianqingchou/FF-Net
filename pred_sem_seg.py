'''
Created on 2023年2月23日

@author: dongzi
'''

import os
import time
from model import get_model, get_loss
from dataset import get_dataset
from dataset.plant_dataset import PlantDataset
import logging
from utils.log_util import LogUtil
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import yaml
from easydict import EasyDict
import torch
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
    
    classifier = get_model(model, num_class=num_class)
    criterion = get_loss(model)
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
            pred_points = []
            
            
            for i, (data, label) in tqdm(enumerate(zip(data, label)), total=len(data)):
                # (B, N, 3)
                data = data[None, ...]
                label = label[:, None]
                
                # (B, 3, N)
                data = torch.from_numpy(data).to(torch.device(conf.device), dtype=torch.float32)
                
                seg_pred, _ = classifier(data.transpose(1, 2))
                seg_pred_choice = np.argmax(seg_pred.cpu().numpy(), axis=-1)
                
                ground_points.append(np.concatenate([data.cpu().detach().squeeze(), label], axis= -1))
                pred_points.append(np.concatenate([data.cpu().detach().squeeze(), seg_pred_choice.squeeze()[:, None]], axis = -1))
                
            
            ground_points = np.concatenate(ground_points, axis=0)
            pred_points = np.concatenate(pred_points, axis=0)
        
        np.savetxt(ground_path, ground_points)
        np.savetxt(pred_path, pred_points)
    log.info('prediction finished...')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-conf', default='configs/pred_conf.yaml')

    args = parser.parse_args()
    
    main(args)

