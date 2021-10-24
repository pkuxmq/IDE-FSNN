import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
import time
import copy
from modules.broyden import broyden, analyze_broyden

import logging
logger = logging.getLogger(__name__)


class SNNIDEModule(nn.Module):

    """ 
    SNN module with implicit differentiation on the equilibrium point in the inner 'Backward' class.
    """

    def __init__(self, snn_func, snn_func_copy):
        super(SNNIDEModule, self).__init__()
        self.snn_func = snn_func
        self.snn_func_copy = snn_func_copy

    def forward(self, u, **kwargs):
        time_step = kwargs.get('time_step', 30)
        threshold = kwargs.get('threshold', 30)
        input_type = kwargs.get('input_type', 'constant')
        solver_type = kwargs.get('solver_type', 'broy')
        leaky = kwargs.get('leaky', None)
        get_all_rate = kwargs.get('get_all_rate', False)

        with torch.no_grad():
            if input_type != 'constant':
                if len(u.size()) == 3:
                    u = u.permute(2, 0, 1)
                else:
                    u = u.permute(4, 0, 1, 2, 3)

            if get_all_rate:
                r_list = self.snn_func.snn_forward(u, time_step, output_type='all_rate', input_type=input_type)
                return r_list

            # To ensure proper BN calculation, we fix the statistics of BN in network_s, 
            # while calculate that for network_x (w.r.t average firing rate), 
            # but set back to the original running_mean and running_var after computation
            self.snn_func.set_bn_mode_s('eval')
            self.snn_func.save_bn_statistics_x()
            z1_out = self.snn_func.snn_forward(u, time_step, input_type=input_type)
            if self.training:
                self.snn_func.set_bn_mode_s('train')
                # remove the influence of running_mean and running_var of network_x
                self.snn_func.restore_bn_statistics_x()

        if self.training:
            if input_type != 'constant':
                if leaky == None:
                    u = torch.mean(u, dim=0)
                else:
                    leaky_ = 1.
                    u_ = u[time_step - 1]
                    for i in range(time_step):
                        leaky_ *= leaky
                        u_ += u[time_step - 2 - i] * leaky_
                    u_ /= (1 - leaky_ * leaky) / (1 - leaky)
                    u = u_

            # BN statistics will be updated here
            z1_out_ = self.snn_func.equivalent_func(z1_out, u)

            self.snn_func_copy.copy(self.snn_func)
            self.snn_func_copy.set_bn_mode_s('eval')

            z1_out = self.Replace.apply(z1_out_, z1_out)

            # change the dimension of z1_out to be consistent with the solver
            sizes = z1_out.size()
            B = z1_out.size(0)
            z1_out = z1_out.reshape(B, -1, 1)

            z1_out = self.Backward.apply(self.snn_func_copy, z1_out, u, sizes, threshold, solver_type)

            # change back the dimension
            z1_out = torch.reshape(z1_out, sizes)

            self.snn_func_copy.set_bn_mode_s('train')

        return z1_out

    class Replace(Function):
        @staticmethod
        def forward(ctx, z1, z1_r):
            return z1_r

        @staticmethod
        def backward(ctx, grad):
            return (grad, grad)

    class Backward(Function):
        """
        A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
        in the backward pass.

        """
        @staticmethod
        def forward(ctx, snn_func_copy, z1, u, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.snn_func = snn_func_copy
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            torch.cuda.empty_cache()

            # grad should have dimension (bsz x d_model x seq_len) to be consistent with the solver
            bsz, d_model, seq_len = grad.size()
            grad = grad.clone()
            z1, = ctx.saved_tensors
            u = ctx.u
            args = ctx.args
            sizes, threshold, solver_type = args[-3:]

            snn_func = ctx.snn_func
            z1_temp = z1.clone().detach().requires_grad_()
            u_temp = u.clone().detach()

            def infer_from_vec(z, u):
                # change the dimension of z
                B = sizes[0]
                z_in = torch.reshape(z, sizes)
                return (snn_func.equivalent_func(z_in, u) - z_in).reshape(B, -1, 1)

            with torch.enable_grad():
                y = infer_from_vec(z1_temp, u_temp)

            def g(x):
                y.backward(x, retain_graph=True)   # Retain for future calls to g
                res = z1_temp.grad.clone().detach() + grad
                z1_temp.grad.zero_()
                return res

            if solver_type == 'broy':
                eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
                dl_df_est = torch.zeros_like(grad)

                result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
                dl_df_est = result_info['result']
                nstep = result_info['nstep']
                lowest_step = result_info['lowest_step']
            else:
                dl_df_est = grad
                for i in range(threshold):
                    dl_df_est = (dl_df_est + g(dl_df_est)) / 2.
            
            if threshold > 30:
                torch.cuda.empty_cache()

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            return (None, dl_df_est, None, *grad_args)
