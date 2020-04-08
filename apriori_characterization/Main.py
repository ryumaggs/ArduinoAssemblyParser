import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import csv
import argparse
import sys
import time
import os
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Logger import SummaryPrint
from torch.autograd import Variable
from torchvision import datasets, transforms
import misc
from misc import bcolors
from DataLoaders import PFPSampler,PFPSampler2

from Wrapper import Wrapper

from Networks import AutoConvNetwork, ClassConvNetwork, LatentAutoConvNetwork

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', metavar='N', type=str, help='name of run')
    parser.add_argument('gpu_id', metavar='G', type=str, help='which gpu to use')

    parser.add_argument('network_type', metavar='N', type=str,default='AEConv', help='network type: AEConv, ClassConv, LSTM')
    parser.add_argument('--multi_net', metavar = 'G', type=int, default=1,help='number of networks')
    parser.add_argument('--ryu_testing', action='store_true',help='for yu')
    parser.add_argument('--ryu_datagen',action='store_true',help='gen data')
    parser.add_argument('--ryu_datagen_train',action='store_false',help='gen train?')
    parser.add_argument('--class_conv',action='store_true',help='class_conv')

    parser.add_argument('--print_network', action='store_true', help='print_network for debugging')
    parser.add_argument('--test', action='store_true', help='test')

    parser.add_argument('--fit_curve', action='store_true', help='fit gaussian?')
    parser.add_argument('--checkpoint_every', type=int, default=10, help='checkpoint every n epochs')
    parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint with same name')
    parser.add_argument('--resume', action='store_true', help='resume from epoch we left off of when loading')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to load')


    parser.add_argument('--epochs', metavar='N', type=int, help='number of epochs to run for', default=3000)

    parser.add_argument('--batch_size', metavar='bs', type=int, default=512, help='batch size')
    parser.add_argument('--lr', metavar='lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--rmsprop', action='store_true', help='use rmsprop optimizer')
    parser.add_argument('--sgd', action='store_true', help='use sgd optimizer')
    parser.add_argument('--l2_reg', metavar='lr', type=float, help='learning rate', default=0.0)

    network_types={'AEConv':AutoConvNetwork,'ClassConv':ClassConvNetwork,'latentConv':LatentAutoConvNetwork}
    network_type=args[2]

    network_class=network_types[network_type]
    PFPSampler.add_args(parser)
    PFPSampler2.add_args(parser)
    network_class.add_args(parser)
    args = parser.parse_args(args)
    #print(not args.test)
    #exit(1)
    data_loader = PFPSampler(args, train=not args.test)
    if args.gpu_id == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
    else:
        print(bcolors.OKBLUE + 'Using GPU' + str(args.gpu_id) + bcolors.ENDC)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device('cuda')

    for x, y in enumerate(data_loader):
        print(x, y),

    run_wrapper = Wrapper(args, network_class, data_loader, device,True)
    if args.test:
        if args.ryu_testing:
            run_wrapper.ryu_test_procedure()
        else:
            run_wrapper.test()
    else:
        run_wrapper.train()

if __name__ == "__main__":
    main(sys.argv[1:])



