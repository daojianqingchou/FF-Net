'''
Created on 2023年2月28日

@author: dongzi
'''

import torch.nn as nn
import torch
import sys
sys.path.append('../')
from utils.lovasz_losses import lovasz_softmax

from voxel_utils import VoxelInit, ResBlock, DownBlock, UpBlock


def get_model(point_dims, voxel_dims, num_class, spatial_shape):
    return FusionNet(point_dims, voxel_dims, num_class, spatial_shape)
    
def get_loss(weight):
    return Loss(weight)

class FusionNet(nn.Module):
    def __init__(self, point_dims, voxel_dims, num_class, spatial_shape)->None:
        nn.Module.__init__(self)

        if isinstance(spatial_shape, int):
            self.spatial_shape = [spatial_shape] * 3
        else:
            self.spatial_shape = spatial_shape
        
        
        self.voxel_init = VoxelInit(point_dims[0], point_dims[1], self.spatial_shape)
        
        self.res0 = ResBlock(voxel_dims[0], voxel_dims[0], indice_key='res0')
        
        self.dn1 = DownBlock(voxel_dims[0], voxel_dims[1], indice_key='res1', down_key='dn1')
        self.dn2 = DownBlock(voxel_dims[1], voxel_dims[2], indice_key='res2', down_key='dn2')
        self.dn3 = DownBlock(voxel_dims[2], voxel_dims[3], indice_key='res3', down_key='dn3')
        self.dn4 = DownBlock(voxel_dims[3], voxel_dims[4], indice_key='res4', down_key='dn4')
        
        self.up4 = UpBlock(voxel_dims[4], voxel_dims[3], indice_key='res4', up_key='dn4')
        self.up3 = UpBlock(voxel_dims[3], voxel_dims[2], indice_key='res3', up_key='dn3')
        self.up2 = UpBlock(voxel_dims[2], voxel_dims[1], indice_key='res2', up_key='dn2')
        self.up1 = UpBlock(voxel_dims[1], voxel_dims[0], indice_key='res1', up_key='dn1')
        
        self.res1 = ResBlock(voxel_dims[0], num_class, indice_key='res0')
    
    def forward(self, pc, pc_indices):
        voxel_feat, pc_feat, _ = self.voxel_init(pc, pc_indices)
        
        x_res0 = self.res0(voxel_feat)

        x_d1 = self.dn1(x_res0)
        x_d2 = self.dn2(x_d1)
        x_d3 = self.dn3(x_d2)
        x_d4 = self.dn4(x_d3)
        
        x_u4 = self.up4(x_d4, x_d3)
        x_u3 = self.up3(x_u4, x_d2)
        x_u2 = self.up2(x_u3, x_d1)
        x_u1 = self.up1(x_u2, x_res0)
        
        x_res1 = self.res1(x_u1)
        
        y_voxel = x_res1.dense()
        
        # fuse point feature and voxel feature
        # f_voxel: (B, H, W, D, C)
        f_voxel = x_res1.dense(channels_first=False)
        #print(f'f_voxel_shape:{f_voxel.shape}')
        
        pc_from_voxel = []
        for i, pc_index in enumerate(pc_indices):
            pc_index = pc_index.to(torch.long)
            pc_from_voxel.append(f_voxel[i, pc_index[:, 0], pc_index[:, 1], pc_index[:, 2], :])
        
        # p: (N, C)
        # for p in pc_from_voxel:
        #     print(f'p.shape: {p.shape}, pc_feat.shape:{pc_feat.shape}')
        
        
        return y_voxel, pc_feat
     
     
class Lovasz_Softmax(nn.Module):
        def __init__(self, classes='present', per_class=False, ignore_label=0):
            """
            Multi-class Lovasz-Softmax loss
              classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
              per_image: compute the loss per image instead of per batch
              ignore: void class labels
            """
            nn.Module.__init__(self)
            self.classes = classes
            self.per_class = per_class
            self.ignore_label = ignore_label
            
        def forward(self, output, target):
            return lovasz_softmax(output, target, self.classes, self.per_class, self.ignore_label)   
   
class Loss(nn.Module):
    def __init__(self, weight, ignore_label=255)->None:
        nn.Module.__init__(self)
        self.cross_entropy_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        self.lovasz_fn = Lovasz_Softmax(ignore_label=ignore_label)
        
    def forward(self, pred_point, target_point, pred_voxel, target_voxel):
        # print(pred_point.device, target_point.device, pred_voxel.device, target_voxel.device)
        # print(pred_point.dtype, target_point.dtype, pred_voxel.dtype, target_voxel.dtype)
        # print(pred_point.shape, target_point.shape)
        return self.cross_entropy_fn(pred_point, target_point) + self.lovasz_fn(pred_voxel, target_voxel)
    
    
        
if __name__ == '__main__':
    pass