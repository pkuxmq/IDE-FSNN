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
from modules.snn_modules import SNNIFFunc, SNNLIFFunc, SNNFC

logger = logging.getLogger(__name__)


class SNNIDEFCNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(SNNIDEFCNet, self).__init__()
        self.parse_cfg(cfg)

        self.network_x = SNNFC(self.dim_in, self.dim_hidden, bias=True, dropout=self.dropout, BN=True)
        self.network_s = SNNFC(self.dim_hidden, self.dim_hidden, bias=False, dropout=self.dropout, BN=False)

        if self.leaky == None:
            self.snn_func = SNNIFFunc(self.network_s, self.network_x, vth=self.vth)
        else:
            self.snn_func = SNNLIFFunc(self.network_s, self.network_x, vth=self.vth, leaky=self.leaky)
        self.snn_func_copy = copy.deepcopy(self.snn_func)

        self.network_s._wnorm(norm_range=1.)

        for param in self.snn_func_copy.parameters():
            param.requires_grad_(False)

        self.snn_ide_fc = SNNIDEModule(self.snn_func, self.snn_func_copy)

        self.classifier = nn.Linear(self.dim_hidden, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.dim_in = cfg['MODEL']['dim_in']
        self.dim_hidden = cfg['MODEL']['dim_hidden']
        self.num_classes = cfg['MODEL']['num_classes']
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
            B, D = x.size()
        else:
            B, D, _ = x.size()

        x1 = torch.zeros([B, self.dim_hidden]).to(dev)
        self.snn_func.network_x._reset(x1)
        self.snn_func.network_s._reset(x1)

        z = self.snn_ide_fc(x, time_step=time_step, threshold=threshold, input_type=input_type, solver_type=self.solver, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        input_type = kwargs.get('input_type', 'constant')
        if input_type == 'constant':
            x = x.reshape(B, -1)
        else:
            T = x.size(-1)
            x = x.reshape(B, -1, T)
        z = self._forward(x, **kwargs)
        y = self.classifier(z)

        return y
