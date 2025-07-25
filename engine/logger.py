from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
from Utils.io_utils import write_args, save_config_to_yaml


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.config_dir = os.path.join(self.save_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        file_name = os.path.join(self.config_dir, 'args.txt')
        write_args(args, file_name)

        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a') # 'w')
        if args.tensorboard:
            self.log_info('using tensorboard')
            self.tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
        else:
            self.tb_writer = None
            
    def save_config(self, config):
        save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        print(info)
        info = str(info)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        info = '{}: {}'.format(time_str, info)
        if not info.endswith('\n'):
            info += '\n'
        self.text_writer.write(info)
        self.text_writer.flush()

    def add_scalar(self, **kargs):
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(**kargs)

    def add_image(self, **kargs):
        
        if self.tb_writer is not None:
            self.tb_writer.add_image(**kargs)

    def add_images(self, **kargs):
        
        if self.tb_writer is not None:
            self.tb_writer.add_images(**kargs)

    def close(self):
        self.text_writer.close()
        self.tb_writer.close()

