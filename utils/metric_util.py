'''
Created on 2022年11月19日

@author: 东子
'''
import numpy as np
from sklearn.metrics import confusion_matrix


def cm(grounds, preds, labels):
    if len(preds) > 1:
        cms = []
        for g, p in zip(grounds, preds):
            cms.append(confusion_matrix(g, p, labels=labels))
        
        cms = sum(cms)
    else:
        cms = confusion_matrix(grounds, preds)
        
    return cms


def iou_compute(grounds, preds, labels):
    if len(preds) > 1:
        cms = []
        for g, p in zip(grounds, preds):
            cms.append(confusion_matrix(g, p, labels=labels))
        
        cms = sum(cms)
    else:
        cms = confusion_matrix(grounds, preds, labels=labels)
    
    print(np.diag(cms))
    print(cms.sum(0))
    print(cms.sum(1))
    iou = np.diag(cms) / (cms.sum(0) + cms.sum(1) - np.diag(cms))
    
    return iou

def miou(preds, grounds, labels):
    iou = iou_compute(grounds, preds, labels)
    