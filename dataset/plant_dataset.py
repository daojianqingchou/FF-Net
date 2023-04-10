'''
Created on 2023年2月23日

@author: dongzi
'''

import os
import numpy as np
import h5py
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_dataset(path, split):
    return PlantDataset(path, split=split)

class PlantDataset(Dataset):
    def __init__(self, path, split='train')->None:
        Dataset.__init__(self)
        self.split = split

        self.h5_plant = h5py.File(path, 'r')
        if 'train' == split:
            self.data = self.h5_plant['train']
        elif 'test' == split:
            self.data = self.h5_plant['test']
        else:
            raise Exception('split supports only train or test.')

        

    def __getitem__(self, index):
        key = list(self.data.keys())[index]
        return self.data[key][:, :3], self.data[key][:, 3]


    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def collate_fn(inputs):
        data = []
        label = []
        for i in inputs:
            # data.append(torch.from_numpy(i[0]))
            # label.append(torch.from_numpy(i[1]))
            data.append(i[0])
            label.append(i[1])
            
        return data, label
    
    
if __name__ == '__main__':
    path = 'E:\Dataset\Pheno4D\h5_tomato_2048_align.h5'

    maize_ds = PlantDataset(path)
    
    maize_dl = DataLoader(maize_ds, batch_size=4, shuffle=True) #, collate_fn=PlantDataset.collate_fn)
    
    for data, label in maize_dl:
        print(data)   
        print(label)
        
