'''
Created on 2023年2月28日

@author: dongzi
'''


import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_max


class VoxelInit(nn.Module):
    def __init__(self, in_dim, out_dim, spatial_shape)->None:
        nn.Module.__init__(self)
        self.spatial_shape = spatial_shape
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            
            nn.Linear(16, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
            )
        
    def forward(self, pc, pc_indices):
        pc_indices_batch = []
        
        for batch_id, pc_index in enumerate(pc_indices):
            pc_indices_batch.append(F.pad(pc_index, (1, 0), value=batch_id))
            
        pc = torch.cat(pc, dim=0)
        pc_indices_batch = torch.cat(pc_indices_batch, dim=0)
        
        # shuffle the points
        shuffle_ind = torch.randperm(len(pc_indices_batch))
        # recovery from shuffled points to before
        recover_ind = torch.argsort(shuffle_ind)
        
        pc = pc[shuffle_ind]
        pc_indices_batch = pc_indices_batch[shuffle_ind]
        
        # learning net
        pc_feat = self.net(pc)
        
        uni, inv = torch.unique(pc_indices_batch, return_inverse=True, dim=0)
        voxel_feat = scatter_max(pc_feat, inv, dim=0)[0]
        voxel_ind = uni
        
        # recover the order of points feature
        pc_feat = pc_feat[recover_ind]
        # sparse tensor
        x = spconv.SparseConvTensor(voxel_feat, voxel_ind, self.spatial_shape, len(pc_indices))
        
        return x, pc_feat, pc_indices
            
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, indice_key=None)->None:
        nn.Module.__init__(self)
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_dim, out_dim, kernel_size, padding=1, indice_key=indice_key),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(), 
            )
        
        self.res = spconv.SparseSequential(
            spconv.SubMConv3d(out_dim, out_dim, kernel_size, padding=1, indice_key=indice_key),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(), 
            )
        
    def forward(self, x):
        x_net = self.net(x)
        res = self.res(x_net)
        
        res = res.replace_feature(x_net.features + res.features)
        
        return res
        
        
class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, indice_key, down_key='down')->None:
        nn.Module.__init__(self)
        self.net = spconv.SparseSequential(
            # downsample
            spconv.SparseConv3d(in_dim, out_dim, 3, stride=2, padding=1, indice_key=down_key),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            
            )
        
        self.res = ResBlock(out_dim, out_dim, 3, indice_key=indice_key)
    
    def forward(self, x):
        x = self.net(x)
        x = self.res(x)
        
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, indice_key, up_key='up')->None:
        nn.Module.__init__(self)
        
        self.net = ResBlock(in_dim, in_dim, 3, indice_key)
        self.up = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_dim, out_dim, 3, indice_key=up_key),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )
        
    def forward(self, x, skip):
        x = self.net(x)
        x = self.up(x)
        x = x.replace_feature(x.features + skip.features)
        
        return x
    

if __name__ == '__main__':
    pass