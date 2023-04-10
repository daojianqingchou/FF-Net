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
    
        train_counts = np.zeros(3)
        test_counts = np.zeros(3)
        
        for v in train.values():
            v = v[:, 3]
            uni, counts = np.unique(v, return_counts=True)
            print(uni)
            uni = uni.astype(int)
        
            train_counts[uni] += counts[uni]
            # for i, u in enumerate(uni):      
            #     train_counts[u] += counts[i]
            
        for v in test.values():
            v = v[:, 3]
            uni, counts = np.unique(v, return_counts=True)
            uni = uni.astype(int)
            
            test_counts[uni] += counts[uni]
            # for i, u in enumerate(uni):      
            #     test_counts[u] += counts[i]
    
    train_dis = train_counts / np.sum(train_counts)
    test_dis = test_counts / np.sum(test_counts)
    
    print(f'From file {path}')
    print(f'Distribution of training set: {train_dis}, amount of points: {train_counts}')
    print(f'Distribution of test set: {test_dis}, amount of points: {test_counts}')
    
    return train_dis, test_dis
        
        
'''
Maize:
Distribution of training set: [0.50026469 0.05500345 0.44473187], amount of points: [21136141.  2323891. 18789884.]
Distribution of test set: [0.48554456 0.0740199  0.44043553], amount of points: [7556381. 1151949. 6854363.]

Tomato:
Distribution of training set: [0.49348445 0.04556682 0.46094872], amount of points: [69319828.  6400778. 64749530.]
Distribution of test set: [0.50725918 0.04323272 0.4495081 ], amount of points: [28787991.  2453545. 25510500.]
'''


if __name__ == '__main__':
    maize_path = 'E:\\Dataset\\Pheno4D\\h5_maize.h5'
    tomato_path = 'E:\\Dataset\\Pheno4D\\h5_tomato.h5'
    
    calculate(maize_path)
    calculate(tomato_path)