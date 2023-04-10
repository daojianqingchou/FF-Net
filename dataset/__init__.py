'''
Created on 2023年2月23日

@author: dongzi
'''
import importlib


def get_dataset(name, path, split='train', **nargs):
    return importlib.import_module(name).get_dataset(path, split, **nargs)

if __name__ == '__main__':
    pass