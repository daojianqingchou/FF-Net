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

    try:
        log.info(f'loading from {conf.model_path}')
        checkpoint = torch.load(conf.model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log.info(f'Use pretrain model {classifier}')
    except:
        log.info('No existing model, starting training from scratch...')
        start_epoch = 0


    # dataset    
    s_point = conf.s_points
    data_path = conf.data_path
    test_dataset = get_dataset('plant_dataset', data_path, 'test')

    batch_size = conf.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
        
        
        for i, (data, label) in tqdm(enumerate(test_loader), total=num_batches):
            # data = np.concatenate(data, axis=0)
            # label = np.concatenate(label, axis=0)
            
            data = data.to(torch.device(conf.device), dtype=torch.float32)
            data = data.transpose(1, 2)
            
            label = label.numpy()
            # label = label.to(torch.int8)
            
            seg_pred, _ = classifier(data)
            seg_pred_choice = np.argmax(seg_pred.cpu().numpy(), axis=-1)
            
            total_correct += np.sum(seg_pred_choice == label)
            total_seen += num_batches * label.shape[1]
            
            for i in range(num_class):
                total_seen_class[i] += np.sum(label == i)
                total_correct_class[i] += np.sum((seg_pred_choice == i) & (label == i))
                total_union_class[i] += np.sum((seg_pred_choice == i) | (label == i))
        
        mIoU = np.mean(total_correct_class / (total_union_class + 1e-6))
        log.info(f"model {model} trained on {conf.model_path} are evaluated on {data_path[data_path.rindex(os.sep)+1:data_path.rindex('.')]}")
        log.info(f'eval point accuracy: {total_correct / total_seen}')
        log.info(f'eval point class accuracy: {total_correct_class / total_seen_class}')
        log.info(f'eval class IoU (ground, leaf, stem): {total_correct_class / (total_union_class + 1e-6)}')
        log.info(f'eval avg class IoU: {mIoU}')        
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-conf', default='configs/test_conf.yaml')

    args = parser.parse_args()
    
    main(args)

