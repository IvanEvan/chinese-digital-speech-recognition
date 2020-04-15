#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Evan
from torch.nn import *
import time
import torch


class BasicModule(Module):
    """
    模型封装
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path, map_location):
        self.load_state_dict(torch.load(path, map_location))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(Module):

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        """
        把输入向量 reshape (batch_size, dim_length)
        :param x:
        :return:
        """
        return x.view(x.size(0), -1)
