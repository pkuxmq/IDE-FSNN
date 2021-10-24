import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
import os
sys.path.append('../')
from modules.optimizations import VariationalHidDropout2d, weight_spectral_norm


class SNNFunc(nn.Module):

    def __init__(self, network_s, network_x, vth):
        # network_s is the network for feedback spike, network_x is the network for input x
        super(SNNFunc, self).__init__()
        self.network_s = network_s
        self.network_x = network_x
        self.vth = torch.tensor(vth, requires_grad=False)

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        pass

    def equivalent_func(self, a, x):
        # equivalent fix-point equation
        a_out = (self.network_s(a) + self.network_x(x)) / self.vth
        return torch.clamp(a_out, 0, 1)

    def forward(self, x, time_step):
        return self.snn_forward(x, time_step)

    def copy(self, target):
        self.network_s.copy(target.network_s)
        self.network_x.copy(target.network_x)

    def set_bn_mode_s(self, mode='train'):
        if mode == 'train':
            if self.network_s.BN:
                self.network_s.bn.train()
        else:
            if self.network_s.BN:
                self.network_s.bn.eval()

    def save_bn_statistics(self):
        self.network_x.save_bn_statistics()
        self.network_s.save_bn_statistics()

    def restore_bn_statistics(self):
        self.network_x.restore_bn_statistics()
        self.network_s.restore_bn_statistics()

    def save_bn_statistics_x(self):
        self.network_x.save_bn_statistics()

    def restore_bn_statistics_x(self):
        self.network_x.restore_bn_statistics()


class SNNIFFunc(SNNFunc):

    def __init__(self, network_s, network_x, vth):
        super(SNNIFFunc, self).__init__(network_s, network_x, vth)

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        if input_type == 'constant':
            x1 = self.network_x(x)
        else:
            # if input data are spikes, for BN in the network_x, we calculate the mean and variance during training w.r.t to the average firing rates
            if self.network_x.BN and self.network_x.bn.training:
                with torch.no_grad():
                    x_mean = torch.mean(x, dim=0)
                    x_mean = self.network_x.forward_linear(x_mean)
                    if len(x_mean.shape) == 4:
                        mean = torch.mean(x_mean, dim=(0, 2, 3), keepdim=True)
                        var = torch.var(x_mean, dim=(0, 2, 3), keepdim=True)
                    else:
                        mean = torch.mean(x_mean, dim=0, keepdim=True)
                        var = torch.var(x_mean, dim=0, keepdim=True)
                    var = torch.sqrt(var + 1e-8)
                x1 = self.network_x(x[0], BN_mean_var=[mean, var])
            else:
                x1 = self.network_x(x[0])

        u = x1
        s = (u >= self.vth).float()
        u = u - self.vth * s

        a = s
        if output_type == 'spike_train':
            sizes = s.size()
            shape = [time_step]
            for i in range(len(sizes)):
                shape.append(sizes[i])
            ss = torch.zeros(shape).to(s.device)
            ss[0] = s

        for t in range(time_step - 1):
            if input_type == 'constant':
                u = u + self.network_s(s) + x1
            else:
                if self.network_x.BN and self.network_x.bn.training:
                    u = u + self.network_s(s) + self.network_x(x[t + 1], BN_mean_var=[mean, var])
                else:
                    u = u + self.network_s(s) + self.network_x(x[t + 1])

            s = (u >= self.vth).float()
            u = u - self.vth * s

            a = a + s
            if output_type == 'spike_train':
                ss[t + 1] = s

        if output_type == 'normal':
            return a * 1.0 / time_step
        elif output_type == 'all_rate':
            return [a * 1.0 / time_step]
        else:
            return ss, a * 1.0 / time_step


class SNNLIFFunc(SNNFunc):

    def __init__(self, network_s, network_x, vth, leaky):
        super(SNNLIFFunc, self).__init__(network_s, network_x, vth)
        self.leaky = torch.tensor(leaky, requires_grad=False)

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        if input_type == 'constant':
            x1 = self.network_x(x)
        else:
            # if input data are spikes, for BN in the network, we calculate the mean and variance during training w.r.t to the weighted average firing rates
            if self.network_x.BN and self.network_x.bn.training:
                with torch.no_grad():
                    leaky_ = 1.
                    x_mean = x[time_step - 1]
                    for t in range(time_step - 1):
                        leaky_ *= self.leaky
                        x_mean += x[time_step - 2 - t] * leaky_
                    x_mean /= (1 - leaky_ * self.leaky) / (1 - self.leaky)
                    x_mean = self.network_x.forward_linear(x_mean)
                    if len(x_mean.shape) == 4:
                        mean = torch.mean(x_mean, dim=(0, 2, 3), keepdim=True)
                        var = torch.var(x_mean, dim=(0, 2, 3), keepdim=True)
                    else:
                        mean = torch.mean(x_mean, dim=0, keepdim=True)
                        var = torch.var(x_mean, dim=0, keepdim=True)
                    var = torch.sqrt(var + 1e-8)
                x1 = self.network_x(x[0], BN_mean_var=[mean, var])
            else:
                x1 = self.network_x(x[0])

        u = x1
        s = (u >= self.vth).float()
        u = u - self.vth * s
        # add leaky term here
        u = u * self.leaky

        a = s
        if output_type == 'spike_train':
            sizes = s.size()
            shape = [time_step]
            for i in range(len(sizes)):
                shape.append(sizes[i])
            ss = torch.zeros(shape).to(s.device)
            ss[0] = s
        elif output_type == 'all_rate':
            r = s

        for t in range(time_step - 1):
            if input_type == 'constant':
                u = u + self.network_s(s) + x1
            else:
                if self.network_x.BN and self.network_x.bn.training:
                    u = u + self.network_s(s) + self.network_x(x[t + 1], BN_mean_var=[mean, var])
                else:
                    u = u + self.network_s(s) + self.network_x(x[t + 1])

            s = (u >= self.vth).float()
            u = u - self.vth * s
            # add leaky term here
            u = u * self.leaky

            a = a * self.leaky + s
            if output_type == 'spike_train':
                ss[t + 1] = s
            elif output_type == 'all_rate':
                r = r + s

        if output_type == 'normal':
            return a / ((1 - self.leaky ** time_step) / (1 - self.leaky))
        elif output_type == 'all_rate':
            return [r * 1. / self.time_step]
        else:
            return ss, a / ((1 - self.leaky ** time_step) / (1 - self.leaky))


class SNNFuncMultiLayer(nn.Module):

    def __init__(self, network_s_list, network_x, vth, fb_num=1):
        # network_s_list is a list of networks, the last fb_num ones are the feedback while previous are feed-forward
        super(SNNFuncMultiLayer, self).__init__()
        self.network_s_list = network_s_list
        self.network_x = network_x
        self.vth = torch.tensor(vth, requires_grad=False)
        self.fb_num = fb_num

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        pass

    def equivalent_func_first(self, a, x):
        # equivalent fix-point equation, a is the first layer
        for i in range(len(self.network_s_list) - 1):
            a = torch.clamp((self.network_s_list[i](a)) / self.vth, 0, 1)
        a = torch.clamp((self.network_s_list[-1](a) + self.network_x(x)) / self.vth, 0, 1)
        return a

    def equivalent_func_last(self, a, x):
        # equivalent fix-point equation, a is the last layer
        if self.fb_num > 1:
            for i in range(self.fb_num - 1):
                a = torch.clamp((self.network_s_list[-self.fb_num + i](a)) / self.vth, 0, 1)
        a = torch.clamp((self.network_s_list[-1](a) + self.network_x(x)) / self.vth, 0, 1)

        for i in range(len(self.network_s_list) - self.fb_num):
            a = torch.clamp((self.network_s_list[i](a)) / self.vth, 0, 1)
        return a

    def equivalent_feedforward(self, x):
        for i in range(len(self.network_s_list) - self.fb_num):
            x = torch.clamp(self.network_s_list[i](x) / self.vth, 0, 1)
        return x

    def equivalent_func(self, a, x, f_type='last'):
        if f_type == 'first':
            return self.equivalent_func_first(a, x)
        else:
            return self.equivalent_func_last(a, x)

    def forward(self, x, time_step):
        return self.snn_forward(x, time_step)

    def copy(self, target):
        for i in range(len(self.network_s_list)):
            self.network_s_list[i].copy(target.network_s_list[i])
        self.network_x.copy(target.network_x)

    def set_bn_mode_s(self, mode='train'):
        if mode == 'train':
            for i in range(len(self.network_s_list)):
                if self.network_s_list[i].BN:
                    self.network_s_list[i].bn.train()
        else:
            for i in range(len(self.network_s_list)):
                if self.network_s_list[i].BN:
                    self.network_s_list[i].bn.eval()

    def save_bn_statistics(self):
        self.network_x.save_bn_statistics()
        for i in range(len(self.network_s_list)):
            self.network_s_list[i].save_bn_statistics()

    def restore_bn_statistics(self):
        self.network_x.restore_bn_statistics()
        for i in range(len(self.network_s_list)):
            self.network_s_list[i].restore_bn_statistics()

    def save_bn_statistics_x(self):
        self.network_x.save_bn_statistics()

    def restore_bn_statistics_x(self):
        self.network_x.restore_bn_statistics()


class SNNIFFuncMultiLayer(SNNFuncMultiLayer):

    def __init__(self, network_s_list, network_x, vth, fb_num=1):
        super(SNNIFFuncMultiLayer, self).__init__(network_s_list, network_x, vth, fb_num)

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        if input_type == 'constant':
            x1 = self.network_x(x)
        else:
            # if input data are spikes, for BN in the network, we calculate the mean and variance during training w.r.t to the average firing rates
            if self.network_x.BN and self.network_x.bn.training:
                with torch.no_grad():
                    x_mean = torch.mean(x, dim=0)
                    x_mean = self.network_x.forward_linear(x_mean)
                    if len(x_mean.shape) == 4:
                        mean = torch.mean(x_mean, dim=(0, 2, 3), keepdim=True)
                        var = torch.var(x_mean, dim=(0, 2, 3), keepdim=True)
                    else:
                        mean = torch.mean(x_mean, dim=0, keepdim=True)
                        var = torch.var(x_mean, dim=0, keepdim=True)
                    var = torch.sqrt(var + 1e-8)
                x1 = self.network_x(x[0], BN_mean_var=[mean, var])
            else:
                x1 = self.network_x(x[0])

        u_list = []
        s_list = []
        u1 = x1
        s1 = (u1 >= self.vth).float()
        u1 = u1 - self.vth * s1
        u_list.append(u1)
        s_list.append(s1)

        for i in range(len(self.network_s_list) - 1):
            ui = self.network_s_list[i](s_list[-1])
            si = (ui >= self.vth).float()
            ui = ui - self.vth * si
            u_list.append(ui)
            s_list.append(si)

        af = s_list[0]
        al = s_list[-self.fb_num]
        if output_type == 'all_rate':
            r_list = []
            for s in s_list:
                r_list.append(s)

        for t in range(time_step - 1):
            if input_type == 'constant':
                u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + x1
            else:
                if self.network_x.BN and self.network_x.bn.training:
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + self.network_x(x[t+1], BN_mean_var=[mean, var])
                else:
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + self.network_x(x[t+1])

            s_list[0] = (u_list[0] >= self.vth).float()
            u_list[0] = u_list[0] - self.vth * s_list[0]

            for i in range(len(self.network_s_list) - 1):
                u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i])
                s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                u_list[i + 1] = u_list[i + 1] - self.vth * s_list[i + 1]

            af = af + s_list[0]
            al = al + s_list[-self.fb_num]
            if output_type == 'all_rate':
                for i in range(len(r_list)):
                    r_list[i] = r_list[i] + s_list[i]

        if output_type == 'normal':
            return af * 1.0 / time_step, al * 1.0 / time_step
        elif output_type == 'all_rate':
            for i in range(len(r_list)):
                r_list[i] *= 1.0 / time_step
            return r_list
        elif output_type == 'first':
            return af * 1.0 / time_step
        else:
            return al * 1.0 / time_step


class SNNLIFFuncMultiLayer(SNNFuncMultiLayer):

    def __init__(self, network_s_list, network_x, vth, leaky, fb_num=1):
        super(SNNLIFFuncMultiLayer, self).__init__(network_s_list, network_x, vth, fb_num)
        self.leaky = torch.tensor(leaky, requires_grad=False)

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        if input_type == 'constant':
            x1 = self.network_x(x)
        else:
            # if input data are spikes, for BN in the network, we calculate the mean and variance during training w.r.t to the weighted average firing rates
            if self.network_x.BN and self.network_x.bn.training:
                with torch.no_grad():
                    leaky_ = 1.
                    x_mean = x[time_step - 1]
                    for t in range(time_step - 1):
                        leaky_ *= self.leaky
                        x_mean += x[time_step - 2 - t] * leaky_
                    x_mean /= (1 - leaky_ * self.leaky) / (1 - self.leaky)
                    x_mean = self.network_x.forward_linear(x_mean)
                    if len(x_mean.shape) == 4:
                        mean = torch.mean(x_mean, dim=(0, 2, 3), keepdim=True)
                        var = torch.var(x_mean, dim=(0, 2, 3), keepdim=True)
                    else:
                        mean = torch.mean(x_mean, dim=0, keepdim=True)
                        var = torch.var(x_mean, dim=0, keepdim=True)
                    var = torch.sqrt(var + 1e-8)
                x1 = self.network_x(x[0], BN_mean_var=[mean, var])
            else:
                x1 = self.network_x(x[0])

        u_list = []
        s_list = []
        u1 = x1
        s1 = (u1 >= self.vth).float()
        u1 = u1 - self.vth * s1
        # add leaky term here
        u1 = u1 * self.leaky

        u_list.append(u1)
        s_list.append(s1)
        for i in range(len(self.network_s_list) - 1):
            ui = self.network_s_list[i](s_list[-1])
            si = (ui >= self.vth).float()
            ui = ui - self.vth * si
            # add leaky term here
            ui = ui * self.leaky

            u_list.append(ui)
            s_list.append(si)

        af = s_list[0]
        al = s_list[-self.fb_num]
        if output_type == 'all_rate':
            r_list = []
            for s in s_list:
                r_list.append(s)

        for t in range(time_step - 1):
            if input_type == 'constant':
                u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + x1
            else:
                if self.network_x.BN and self.network_x.bn.training:
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + self.network_x(x[t+1], BN_mean_var=[mean, var])
                else:
                    u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + self.network_x(x[t+1])

            s_list[0] = (u_list[0] >= self.vth).float()
            u_list[0] = u_list[0] - self.vth * s_list[0]
            # add leaky term here
            u_list[0] = u_list[0] * self.leaky

            for i in range(len(self.network_s_list) - 1):
                u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i])
                s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                u_list[i + 1] = u_list[i + 1] - self.vth * s_list[i + 1]
                # add leaky term here
                u_list[i + 1] = u_list[i + 1] * self.leaky

            af = af * self.leaky + s_list[0]
            al = al * self.leaky + s_list[-self.fb_num]
            if output_type == 'all_rate':
                for i in range(len(r_list)):
                    r_list[i] = r_list[i] + s_list[i]

        weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
        if output_type == 'normal':
            return af / weighted, al / weighted
        elif output_type == 'all_rate':
            for i in range(len(r_list)):
                r_list[i] *= 1.0 / time_step
            return r_list
        elif output_type == 'first':
            return af / weighted
        else:
            return al / weighted


class SNNFC(nn.Module):

    def __init__(self, d_in, d_out, bias=False, need_resize=False, sizes=None, dropout=0.0, BN=False):
        super(SNNFC, self).__init__()
        self.fc = nn.Linear(d_in, d_out, bias=bias)
        self.need_resize = need_resize
        self.sizes=sizes
        self.drop = VariationalHidDropout2d(dropout, spatial=False)
        self.BN = BN
        if self.BN:
            self.bn = nn.BatchNorm1d(d_out)

        self._initialize_weights()

    def forward(self, x, BN_mean_var=None):
        if self.need_resize:
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        if self.BN:
            if BN_mean_var == None:
                x = self.bn(x)
            else:
                x = (x - BN_mean_var[0]) / BN_mean_var[1] * self.bn.weight.reshape(1, -1) + self.bn.bias.reshape(1, -1)
        return self.drop(x)

    def forward_linear(self, x):
        if self.need_resize:
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        return x

    def _wnorm(self, norm_range=1.):
        self.fc, self.fc_fn = weight_spectral_norm(self.fc, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'fc_fn' in self.__dict__:
            self.fc_fn.reset(self.fc)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.fc
        m.weight.data.uniform_(-1, 1)
        for i in range(m.weight.size(0)):
            m.weight.data[i] /= torch.norm(m.weight.data[i])
        if m.bias is not None:
            m.bias.data.zero_()

        if self.BN:
            m = self.bn
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def copy(self, target):
        self.fc.weight.data = target.fc.weight.data.clone()
        if self.fc.bias is not None:
            self.fc.bias.data = target.fc.bias.data.clone()
        self.drop.mask = target.drop.mask.clone()

        if self.BN:
            self.bn.weight.data = target.bn.weight.data.clone()
            self.bn.bias.data = target.bn.bias.data.clone()
            self.bn.running_mean.data = target.bn.running_mean.data.clone()
            self.bn.running_var.data = target.bn.running_var.data.clone()

        self.need_resize = target.need_resize
        self.sizes = target.sizes

    def save_bn_statistics(self):
        if self.BN:
            self.bn_running_mean = self.bn.running_mean
            self.bn_running_var = self.bn.running_var

    def restore_bn_statistics(self):
        if self.BN:
            self.bn.running_mean.data = self.bn_running_mean.data.clone()
            self.bn.running_var.data = self.bn_running_var.data.clone()


class SNNConv(nn.Module):

    def __init__(self, d_in, d_out, kernel_size, bias=True, BN=False, stride=1, padding=None, pooling=False, dropout=0.0):
        super(SNNConv, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(d_in, d_out, kernel_size, stride, padding, bias=bias)
        self.BN = BN
        self.pooling = pooling
        if self.BN:
            self.bn = nn.BatchNorm2d(d_out)
        if self.pooling:
            self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.drop = VariationalHidDropout2d(dropout, spatial=False)

        self._initialize_weights()

    def forward(self, x, BN_mean_var=None):
        x = self.conv(x)
        if self.BN:
            if BN_mean_var == None:
                x = self.bn(x)
            else:
                x = (x - BN_mean_var[0]) / BN_mean_var[1] * self.bn.weight.reshape(1, -1, 1, 1) + self.bn.bias.reshape(1, -1, 1, 1)
        if self.pooling:
            x = self.pool(x)
        return self.drop(x)

    def forward_linear(self, x):
        return self.conv(x)

    def _wnorm(self, norm_range=1.):
        self.conv, self.conv_fn = weight_spectral_norm(self.conv, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'conv_fn' in self.__dict__:
            self.conv_fn.reset(self.conv)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.conv
        m.weight.data.uniform_(-1, 1)
        for i in range(m.out_channels):
            m.weight.data[i] /= torch.norm(m.weight.data[i])
        if m.bias is not None:
            m.bias.data.zero_()

        if self.BN:
            m = self.bn
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def copy(self, target):
        self.conv.weight.data = target.conv.weight.data.clone()
        if self.conv.bias is not None:
            self.conv.bias.data = target.conv.bias.data.clone()

        if self.BN:
            self.bn.weight.data = target.bn.weight.data.clone()
            self.bn.bias.data = target.bn.bias.data.clone()
            self.bn.running_mean.data = target.bn.running_mean.data.clone()
            self.bn.running_var.data = target.bn.running_var.data.clone()

        self.drop.mask = target.drop.mask.clone()

    def save_bn_statistics(self):
        if self.BN:
            self.bn_running_mean = self.bn.running_mean
            self.bn_running_var = self.bn.running_var

    def restore_bn_statistics(self):
        if self.BN:
            self.bn.running_mean.data = self.bn_running_mean.data.clone()
            self.bn.running_var.data = self.bn_running_var.data.clone()


class SNNConvTranspose(nn.Module):

    def __init__(self, d_in, d_out, kernel_size=3, bias=False, BN=False, stride=2, padding=1, output_padding=1, dropout=0.0):
        super(SNNConvTranspose, self).__init__()
        self.convT = nn.ConvTranspose2d(d_in, d_out, kernel_size, stride, padding, output_padding, bias=bias)
        self.drop = VariationalHidDropout2d(dropout, spatial=False)
        self.BN = BN
        if self.BN:
            self.bn = nn.BatchNorm2d(d_out)

        self._initialize_weights()

    def forward(self, x, BN_mean_var=None):
        x = self.convT(x)
        if self.BN:
            if BN_mean_var == None:
                x = self.bn(x)
            else:
                x = (x - BN_mean_var[0]) / BN_mean_var[1] * self.bn.weight.reshape(1, -1, 1, 1) + self.bn.bias.reshape(1, -1, 1, 1)
        return self.drop(x)

    def _wnorm(self, norm_range=1.):
        self.convT, self.convT_fn = weight_spectral_norm(self.convT, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'convT_fn' in self.__dict__:
            self.convT_fn.reset(self.convT)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.convT
        m.weight.data.uniform_(-1, 1)
        for i in range(m.out_channels):
            m.weight.data[:, i] /= torch.norm(m.weight.data[:, i])
        if m.bias is not None:
            m.bias.data.zero_()

        if self.BN:
            m = self.bn
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def copy(self, target):
        self.convT.weight.data = target.convT.weight.data.clone()
        if self.convT.bias is not None:
            self.convT.bias.data = target.convT.bias.data.clone()

        if self.BN:
            self.bn.weight.data = target.bn.weight.data.clone()
            self.bn.bias.data = target.bn.bias.data.clone()
            self.bn.running_mean.data = target.bn.running_mean.data.clone()
            self.bn.running_var.data = target.bn.running_var.data.clone()

        self.drop.mask = target.drop.mask.clone()

    def save_bn_statistics(self):
        if self.BN:
            self.bn_running_mean = self.bn.running_mean
            self.bn_running_var = self.bn.running_var

    def restore_bn_statistics(self):
        if self.BN:
            self.bn.running_mean.data = self.bn_running_mean.data.clone()
            self.bn.running_var.data = self.bn_running_var.data.clone()

