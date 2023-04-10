'''
Created on 2023年2月23日

@author: dongzi
'''

import os
import numpy as np
import h5py
import torch
from numba import jit

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_dataset(path, split, spatial_shape=100):
    return PlantVoxelDataset(path, split=split, spatial_shape=spatial_shape)

class PlantVoxelDataset(Dataset):
    def __init__(self, path, split='train', spatial_shape=200)->None:
        Dataset.__init__(self)
        self.split = split
        
        self.h5_plant = h5py.File(path, 'r')
        if 'train' == split:
            self.data = self.h5_plant['train']
        elif 'test' == split:
            self.data = self.h5_plant['test']
        else:
            raise Exception('split supports only train or test.')

        # v_size (x, y, z)
        if isinstance(spatial_shape, int):
            self.spatial_shape = [spatial_shape] * 3
        else:
            self.spatial_shape = spatial_shape
        

    def __getitem__(self, index):
        key = list(self.data.keys())[index]
        pc, label = self.data[key][:, :3], self.data[key][:, 3]
        
        label = label.astype(np.int16)
        
        min_bound = np.min(pc, axis=0, keepdims=True)
        max_bound = np.max(pc, axis=0, keepdims=True)
        
        interval = (max_bound - min_bound) / self.spatial_shape  + 1e-6
        pc_indices = ((pc - min_bound) // interval).astype(np.int16)
        
        # point coordinate with respect to center of voxel which the point reside in.
        pc_feat1 = pc - (pc_indices + 0.5) * interval
        # point coordinate with respect to the border of voxel in which the point reside.
        pc_feat2 = pc - pc_indices * interval 
        pc_feats = np.concatenate([pc, pc_feat1, pc_feat2], axis=-1)
        
        pc_label_pair = np.concatenate((pc_indices, label[:, None]), axis=-1)
        # pc_label_pair = pc_label_pair[np.lexsort(pc_label_pair[:, :3].T)]
        pc_label_pair = pc_label_pair[np.lexsort((pc_label_pair[:, 0], pc_label_pair[:, 1], pc_label_pair[:, 2]), 0)]
        
        voxel_label = np.ones(self.spatial_shape, np.int8) * 255
        voxel_label = voxel_label_gathering(voxel_label, pc_label_pair)
        voxel_indices = np.indices(self.spatial_shape, dtype=np.int16)
        
        
        return voxel_indices, voxel_label, pc_feats, pc_indices, label 
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def collate_fn(inputs):
        voxel_indices = []
        voxel_label = []
        pc = []
        pc_indices = []
        label = []

        for i in inputs:
            voxel_indices.append(i[0])
            voxel_label.append(i[1])
            pc.append(i[2].astype(np.float32))
            pc_indices.append(i[3].astype(np.int32))
            label.append(i[4])
            
            
        return voxel_indices, voxel_label, pc, pc_indices, label


@jit(nopython=True)
def voxel_label_gathering(voxel_label, pc_label_pair):
    """obtain the voxel labels by selecting the points label whose count is the most."""
    label_size = 3
    counter = np.zeros(label_size, np.int16)
    cur_point = pc_label_pair[0, :3]
    for i in range(pc_label_pair.shape[0]):
        if np.any(cur_point != pc_label_pair[i, :3]):
            voxel_label[cur_point[0], cur_point[1], cur_point[2]] = np.argmax(counter)
            # print(i, ' -- ', [cur_point[0], cur_point[1], cur_point[2]], ' -- ', np.sum(counter), '*',  np.argmax(counter))
            counter = np.zeros(label_size, np.int16)
            cur_point = pc_label_pair[i, :3]
        counter[pc_label_pair[i, -1]] += 1
    voxel_label[cur_point[0], cur_point[1], cur_point[2]] = np.argmax(counter)
    return voxel_label   
 
def voxelize(pc, label, spatial_shape):
    label = label.astype(np.int16)
    min_bound = np.min(pc, axis=0, keepdims=True)
    max_bound = np.max(pc, axis=0, keepdims=True)
    
    interval = (max_bound - min_bound) / spatial_shape  + 1e-6
    pc_indices = ((pc - min_bound) // interval).astype(np.int16)
    
    # point coordinate with respect to center of voxel which the point reside in.
    pc_feat1 = pc - (pc_indices + 0.5) * interval
    # point coordinate with respect to the border of voxel in which the point reside.
    pc_feat2 = pc - pc_indices * interval 
    pc_feats = np.concatenate([pc, pc_feat1, pc_feat2], axis=-1)
    
    pc_label_pair = np.concatenate((pc_indices, label[:, None]), axis=-1)
    # pc_label_pair = pc_label_pair[np.lexsort(pc_label_pair[:, :3].T)]
    pc_label_pair = pc_label_pair[np.lexsort((pc_label_pair[:, 0], pc_label_pair[:, 1], pc_label_pair[:, 2]), 0)]
    
    voxel_label = np.ones(spatial_shape, np.int8) * 255
    voxel_label = voxel_label_gathering(voxel_label, pc_label_pair)
    voxel_indices = np.indices(spatial_shape, dtype=np.int16)
    
    
    return voxel_indices, voxel_label, pc_feats, pc_indices, label 
 
    
if __name__ == '__main__':
    path = 'E:\Dataset\Pheno4D\h5_maize.h5'

    maize_ds = PlantVoxelDataset(path, spatial_shape=80)
    
    maize_dl = DataLoader(maize_ds, batch_size=4, shuffle=True, collate_fn=PlantVoxelDataset.collate_fn)
    
    for voxel_indices, voxel_label, pc, pc_indices, label in maize_dl:
        print(voxel_indices[0].shape)   
        print(voxel_label[0].shape)
        print(pc[0].shape)
        print(pc_indices[0].shape)
        print(label[0].shape)
        
