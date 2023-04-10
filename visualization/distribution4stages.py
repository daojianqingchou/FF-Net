'''
Created on 2023年2月27日

@author: dongzi
'''
import os
import h5py
import numpy as np


def calculate(path):
    r"""path: the path of hdf5 file."""
    if not os.path.exists(path):
        raise Exception(f'These not exist file {path}.')
    with h5py.File(path) as h5:
        train = h5['train']
        test = h5['test']
    
        amounts = {}
        count = 7 # the count of plants
        
        for key in train.keys():
            k_name = key.split('_')[1]
            if k_name not in amounts:
                amounts[k_name] = 0
                
            amounts[k_name] += len(train[key])
            print(f'{key}:{len(train[key])}')
            
        for key in test.keys():
            k_name = key.split('_')[1]

            amounts[k_name] += len(test[key])
            
        print(f'From file {path}')
        print(f'average amounts of different stages:')
        print(f'stages: {list(amounts.keys())}')
        print(f'amounts:{np.asarray(list(amounts.values())) // count}')
    
    return amounts
        
        
'''
Maize:
average amounts of different stages:
stages: ['0313', '0315', '0317', '0319', '0321', '0324', '0325']
amounts:[1530508 1570279  737884  819289 1023204 1245837 1331940]

Tomato:
average amounts of different stages:
stages: ['0305', '0307', '0309', '0311', '0313', '0315', '0317', '0319', '0321', '0324', '0325']
amounts:[2012720 2760514 2515615 2344358 2943827 2158687 2335343 1899911 1878429 3683278 3641908]
'''


if __name__ == '__main__':
    maize_path = 'E:\\Dataset\\Pheno4D\\h5_maize.h5'
    tomato_path = 'E:\\Dataset\\Pheno4D\\h5_tomato.h5'
    
    calculate(maize_path)
    calculate(tomato_path)