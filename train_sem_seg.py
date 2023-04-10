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
    criterion = get_loss(model)
    try:
        checkpoint = torch.load(os.path.join(args.log_dir, 'checkpoints/best_model.pth'))
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
    s_point = conf.s_point
    data_path = conf.data_path
    train_dataset = get_dataset('plant_dataset', data_path, 'train')
    test_dataset = get_dataset('plant_dataset', data_path, 'test')

    batch_size = conf.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
        
        for i, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            # data = torch.concat(data, dim=0)
            # label = torch.concat(data, dim=0)
            # print(data.shape, label.shape)
            
            # data = np.concatenate(data, axis=0)
            # label = np.concatenate(label, axis=0)
            # data = data[None, ...]
            
            data[:, :, :3] = torch.from_numpy(provider.rotate_point_cloud_z(data[:, :, :3]))
            
            # points = data.numpy()
            # data[:, :, :3] = provider.rotate_point_cloud_z(data[:, :, :3])
            # points = torch.Tensor(points)
            # points, target = points.float().cuda(), target.long().cuda()
            # points = points.transpose(2, 1)
            
            # data = torch.from_numpy(data).float().transpose(2, 1)
            data = data.float().transpose(2, 1)
            # label = torch.from_numpy(label)
            
            data = data.to(torch.device(conf.device))
            label = label.to(conf.device, dtype=torch.long)
            # label = label.astype(np.int8)

            seg_pred, tran_feat = classifier(data)
            seg_pred = eo.rearrange(seg_pred, 'b n c -> b c n')
            loss = criterion(seg_pred, label, tran_feat, None)
            
            loss.backward()
            optimizer.step()
            
            seg_pred_choice = np.argmax(seg_pred.cpu().detach().numpy(), -1)
            label = label.cpu().detach().numpy()
            
            total_correct += np.sum(label == seg_pred_choice)
            total_seen += num_batches * label.shape[1]
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
            
            for i, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
            log.info(f'eval point accuracy: {total_correct / total_seen}')
            log.info(f'eval point class accuracy: {total_correct_class / total_seen_class}')
            log.info(f'eval class IoU (ground, leaf, stem): {total_correct_class / (total_union_class + 1e-6)}')
            log.info(f'eval avg class IoU: {mIoU}')        
            
            
            if mIou > best_iou:
                state_dir = os.path.join(log_dir, 'models')
                if not os.path.exists(state_dir):
                    os.makedirs(state_dir)
                    
                model_path = os.path.join(state_dir, f'best_model_{conf.model}_{data_name}.pth')
                
                bset_iou = mIoU
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

