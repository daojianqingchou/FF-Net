'''
Created on 2023年2月23日

@author: dongzi
'''

import importlib
__all__ = ['get_model', 'get_loss']
# MODELS = {}

def get_model(name, **args):
    return importlib.import_module(name).get_model(**args)

def get_loss(name, **args):
    return importlib.import_module(name).get_loss(**args)