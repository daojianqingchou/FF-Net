'''
Created on 2022年11月12日

@author: 东子
'''

import logging

class LogUtil:
    def __init__(self, path):
        self.path = path
        
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler(path, encoding='utf-8')
        so_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        so_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(so_handler)
        
        self.logger.setLevel(logging.INFO)
        
        
    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)


if __name__ == '__main__':
    pass