import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import copy
sys.path.append("../")
from modules.snnide_module import SNNIDEModule
from modules.snn_modules import SNNIFFunc, SNNLIFFunc, SNNConv

logger = logging.getLogger(__name__)


class SNNIDEConvNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SNNIDEConvNet, self).__init__()
        self.parse_cfg(cfg)

        self.network_x = SNNConv(self.c_in, self.c_hidden, self.kernel_size_x, bias=True, BN=True, stride=self.stride_x, padding=self.padding_x, dropout=self.dropout, pooling=self.pooling_x)
        self.network_s = SNNConv(self.c_hidden, self.c_hidden, self.kernel_size_s, bias=False, BN=False, stride=1, dropout=self.dropout)

        if self.leaky == None:
            self.snn_func = SNNIFFunc(self.network_s, self.network_x, vth=self.vth)
        else:
            self.snn_func = SNNLIFFunc(self.network_s, self.network_x, vth=self.vth, leaky=self.leaky)
        self.snn_func_copy = copy.deepcopy(self.snn_func)

        self.network_s._wnorm(norm_range=1.)

        for param in self.snn_func_copy.parameters():
            param.requires_grad_(False)

        self.snn_ide_conv = SNNIDEModule(self.snn_func, self.snn_func_copy)

        self.classifier = nn.Linear(self.h_hidden * self.w_hidden * self.c_hidden, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.c_in = cfg['MODEL']['c_in']
        self.c_hidden = cfg['MODEL']['c_hidden']
        self.h_hidden = cfg['MODEL']['h_hidden']
        self.w_hidden = cfg['MODEL']['w_hidden']
        self.num_classes = cfg['MODEL']['num_classes']
        self.kernel_size_x = cfg['MODEL']['kernel_size_x']
        self.stride_x = cfg['MODEL']['stride_x']
        self.padding_x = cfg['MODEL']['padding_x']
        self.pooling_x = cfg['MODEL']['pooling_x'] if 'pooling_x' in cfg['MODEL'].keys() else False
        self.kernel_size_s = cfg['MODEL']['kernel_size_s']
        self.threshold = cfg['MODEL']['threshold']
        self.time_step = cfg['MODEL']['time_step']
        self.vth = cfg['MODEL']['vth']
        self.dropout = cfg['MODEL']['dropout'] if 'dropout' in cfg['MODEL'].keys() else 0.0
        self.leaky = cfg['MODEL']['leaky'] if 'leaky' in cfg['MODEL'].keys() else None
        if 'OPTIM' in cfg.keys() and 'solver' in cfg['OPTIM'].keys():
            self.solver = cfg['OPTIM']['solver']
        else:
            self.solver = 'broy'

    def _forward(self, x, **kwargs):
        threshold = kwargs.get('threshold', self.threshold)
        time_step = kwargs.get('time_step', self.time_step)
        input_type = kwargs.get('input_type', 'constant')
        leaky = kwargs.get('leaky', self.leaky)
        dev = x.device

        if input_type == 'constant':
            B, C, H, W = x.size()
        else:
            B, C, H, W, _ = x.size()

        x1 = torch.zeros([B, self.c_hidden, self.h_hidden, self.w_hidden]).to(dev)
        self.snn_func.network_x._reset(x1)
        self.snn_func.network_s._reset(x1)

        z = self.snn_ide_conv(x, time_step=time_step, threshold=threshold, input_type=input_type, solver_type=self.solver, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        z = self._forward(x, **kwargs)
        z = z.reshape(B, -1)
        y = self.classifier(z)

        return y
