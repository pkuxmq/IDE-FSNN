from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from models.snn_ide_conv_multilayer import SNNIDEConvMultiLayerNet

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 SNN Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--path', default='./data', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# SNN settings
parser.add_argument('-t', '--time_step', default=100, type=int)
parser.add_argument('--vth', default=2., type=float)
parser.add_argument('--leaky', default=0.99, type=float, help='the leaky term for LIF model, set 1. for IF model')
# Optimization options
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# leaky term
if args.leaky >= 1. or args.leaky <= 0.:
    args.leaky = None

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


def main():

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    testset = dataloader(root=args.path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model SNN_Conv")
    in_planes = 3
    out_planes = num_classes
    in_pixels = 32
    vth = torch.tensor(args.vth).cuda()

    config = {}
    config['MODEL'] = {}
    config['MODEL']['c_in'] = in_planes
    config['MODEL']['c_hidden'] = 96
    config['MODEL']['num_classes'] = num_classes
    config['MODEL']['kernel_size_x'] = 3
    config['MODEL']['stride_x'] = 2
    config['MODEL']['padding_x'] = 1
    config['MODEL']['pooling_x'] = False
    config['MODEL']['c_s1'] = 256
    config['MODEL']['c_s2'] = 384
    config['MODEL']['c_s3'] = 384
    config['MODEL']['c_s4'] = 256
    config['MODEL']['kernel_size_s'] = 3
    config['MODEL']['h_hidden'] = 8
    config['MODEL']['w_hidden'] = 8
    config['MODEL']['dropout'] = 0.0
    config['MODEL']['threshold'] = 30
    config['MODEL']['time_step'] = args.time_step
    config['MODEL']['vth'] = args.vth
    config['MODEL']['leaky'] = args.leaky
    config['OPTIM'] = {}
    config['OPTIM']['solver'] = 'broy'
    model = SNNIDEConvMultiLayerNet(config)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    # Resume
    title = 'cifar-' + 'SNN_Conv'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    print('\nEvaluation only')
    test_loss, test_acc = test(testloader, model, criterion, use_cuda, args.time_step, num_classes)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    return


def test(testloader, model, criterion, use_cuda, time_step, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    r_list = [0., 0., 0., 0., 0.]
    total_r = 0.
    sample_num = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # get firing rates
        batch_r_list = model.module.get_all_rate(inputs)
        total_dims = 0
        batch_r = 0.
        for i in range(len(batch_r_list)):
            r_list[i] += torch.sum(torch.mean(batch_r_list[i], dim=1))
            batch_r += torch.sum(batch_r_list[i])
            total_dims += batch_r_list[i].shape[1]
            sample_num += batch_r_list[i].shape[0]
        total_r += batch_r / total_dims

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    for i in range(len(r_list)):
        print('layer ' + str(i + 1) + ' average firing rate is {:.4f}'.format(r_list[i] / sample_num))
    print('total average firing rate is {:.4f}'.format(total_r / sample_num))

    return (losses.avg, top1.avg)

if __name__ == '__main__':
    main()
