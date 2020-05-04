import torch
import torch.utils.data
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import numpy as np
import os
import csv
import time
import gzip
from scipy.optimize import curve_fit
from Logger import SummaryPrint
from misc import bcolors
import misc
import matplotlib.pyplot as plt
from DataLoaders import PFPSampler

class Wrapper(object):
    def __init__(self, args, network_class, data_loader, device,auto,num_net = 1):
        super(Wrapper, self).__init__()
        self.args = args
        self.device = device
        self.window_size = args.window_size
        self.ctr = 0
        self.nfp = 0
        self.epoch = 0
        self.num_norm = 0
        self.num_ana = 0
        self.norm_limit = 2000
        self.ana_limit = 2000
        self.norm_error = []
        self.ana_error = []
        self.auto=auto
        self.fit_curve = args.fit_curve
        self.gen_train = args.ryu_datagen_train
        self.class_conv = args.class_conv
        self.num_net = args.multi_net
        self.network_list = []
        self.sumPrint_list = []
        self.validSumPrint_list = []
        self.networks_trained = []
        self.ryu_test = False
        print(auto,auto,auto,auto)
        self.ckpt_dir = 'ckpts/{0}'.format(args.run_name)
        if not os.path.isdir('ckpts'): #this is making the ckpts folder
            os.mkdir('ckpts')

        if not os.path.isdir(self.ckpt_dir): #this is making the run ckpt folder
            os.makedirs(self.ckpt_dir)

        print('Creating Network')
        self.data_loader = data_loader
        x,y = next(iter(self.data_loader))
        x_size = x.size()
        if self.num_net > 1: #if parameter is defined, will now generate a list of networks

            self.network = network_class(args,x_size)
            self.network_list.append(self.network)
            self.sumPrint = SummaryPrint(args,self.network.loss_names(), self.ckpt_dir, 'train' if not args.test else 'test')
            self.sumPrint_list.append(self.sumPrint)
            self.validSumPrint_list.append(SummaryPrint(args,self.network.loss_names(),self.ckpt_dir, 'valid', color=bcolors.OKBLUE))
            self.validSumPrint = self.validSumPrint_list[0]
            i = 1
            while i < self.num_net:
                tempnet = network_class(args,x_size)
                self.network_list.append(tempnet) #not sure if i can prematurely send all of them to device
                self.sumPrint_list.append(SummaryPrint(args,self.network_list[i].loss_names(),self.ckpt_dir,'train' if not args.test else 'test'))
                self.validSumPrint_list.append(SummaryPrint(args,self.network_list[i].loss_names(), self.ckpt_dir, 'valid', color=bcolors.OKBLUE))
                i += 1
            #send all the networks to the device
            for i in range(len(self.network_list)):
                self.network_list[i] = self.network_list[i].to(device)
        else:
            self.network = network_class(args, x_size)
            if args.print_network:
                print(self.network)
            #will need to make this into a list
            self.sumPrint = SummaryPrint(args,self.network.loss_names(), self.ckpt_dir, 'train' if not args.test else 'test')
            #temporarily disable validation for multi-network network
            if not args.test:
                self.validSumPrint = SummaryPrint(args,self.network.loss_names(), self.ckpt_dir, 'valid', color=bcolors.OKBLUE)

            print(bcolors.OKBLUE + 'Moving to specified device' + bcolors.ENDC)
            self.network = self.network.to(device)
            cudnn.benchmark = True
        if self.args.rmsprop:
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=args.lr, weight_decay=self.args.l2_reg)
        elif self.args.sgd:
            self.optimizer = optim.SGD(self.network.parameters(), lr=args.lr, momentum=0.9, weight_decay=self.args.l2_reg)
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr, weight_decay=self.args.l2_reg)
            

    def load(self, resume=False):
        if self.args.checkpoint is None:
            print(bcolors.OKBLUE + 'Loading Checkpoint: ' + self.args.run_name + bcolors.ENDC)
            checkpoint = torch.load(self.ckpt_dir+"/ckpt.pth")
        else:
            print(bcolors.OKBLUE + 'Loading Checkpoint: ' + self.args.checkpoint + bcolors.ENDC)
            checkpoint = torch.load("ckpts/%s/ckpt.pth" % self.args.checkpoint)
        self.network.load_state_dict(checkpoint['network'])
        if resume:
            self.optimizer.load_state_dict(checkpoint['opt'])
            self.epoch = checkpoint['epoch']
        print(bcolors.OKBLUE + 'Finished Loading Checkpoint ' + bcolors.ENDC)

    def save(self):
        print(bcolors.OKBLUE + 'Saving Checkpoint: ' + self.args.run_name + bcolors.ENDC)
        torch.save({
            'network':self.network.state_dict(),
            'opt':self.optimizer.state_dict(),
            'epoch':self.epoch+1,
            'args':self.args,
            }, self.ckpt_dir+"/ckpt.pth")

    def _iter(self, x, y, sumPrint, backwards=True):
        x = x[:,0,:]
        x = x.view(x.shape[0],1,x.shape[1])
        x, y = x.to(self.device), y.to(self.device)
        y_bar = self.network(x)
        loss_l = self.network.loss(x,y,y_bar)
	if backwards == False:
		batch = 0
		while(batch < 512):
			truth = y[batch].cpu().numpy()
			guess = y_bar[batch, 0].detach().cpu().numpy()
			print("truth: ", truth, " || guess: ", guess)
			batch += 1
        if backwards == False and self.args.ryu_testing == True:
            batch = 0
            batch_test = []
            while batch < 512:
                label = y[batch].item()

                if label == 2 and self.num_norm > self.norm_limit:
                    return [l.data.item() for l in loss_l]
                if label == 1 and self.num_ana > self.ana_limit:
                    return [l.data.item() for l in loss_l]
                truth = y[batch,:,:].cpu().numpy()
                batch_test.append(truth)

                guess = y_bar[batch,:,:].detach().cpu().numpy()
                l2 = np.linalg.norm(truth-guess)
		print("guess: ", guess, " truth ", truth)
                if label == 2:
                    #print('**************', label)
                    #print('**************', l2)
                    self.norm_error.append(l2)
                    self.num_norm += 1             
                elif label == 1:
                    print("-------------", l2)
                    self.ana_error.append(l2)
                    self.num_ana += 1
                batch += 1
        print("")
        """for i in range(len(batch_test)):
            j = i
            while j < len(batch_test):
                #print(batch_test[i])
                print("Norm error between: ", i, j, " is", np.linalg.norm(batch_test[i]-batch_test[j]))
                j+= 1
            exit(1)"""
        if backwards:
            loss_l = self.network.loss(x,y,y_bar)
            self.optimizer.zero_grad()
            loss_l[0].backward()
            self.optimizer.step()
        return [l.data.item() for l in loss_l]

    def gen_training_testing(self, x, y,training=True): #LEFT OFF HERES
        x_copy = torch.Tensor.numpy(x)
        x_copy = x_copy.tolist()
        i = 0
        while(i < len(x_copy)):
            sample = x_copy[i]
            truth = y[i].item()
            
            strii = ','.join(str(i) for i in sample) + ',' + str(truth) + '\n'
            #print(truth)
            wrote = False
            if training:
                if (self.num_norm < self.norm_limit and truth == 0) or (truth != 0 and self.num_ana < self.ana_limit):
                    with open('./trainingData.txt.gz','ab') as f:
                        f.write(strii)
                    wrote = True
            else:
                if (self.num_norm < self.norm_limit and truth == 0) or (truth != 0 and self.num_ana < self.ana_limit):
                    with open('./testingData.txt.gz','ab') as f:
                        f.write(strii)
                    wrote = True
            i += 1
            if wrote == True:
                if truth == 0:
                    self.num_norm += 1
                else:
                    self.num_ana += 1
            #print(self.num_ana, self.num_norm)
            if self.num_norm > self.norm_limit and self.num_ana > self.ana_limit:
                break
        return 0

    def _ryu_iter(self, x, y, sumPrint, backwards=True): 
        x_copy = torch.Tensor.numpy(x)
        x_copy = x_copy.tolist()
        #print("---------------", len(x_copy))
        
        x,y = x.to(self.device), y.to(self.device)
        y_bar = self.network(x) 
        loss_l = self.network.loss(x, y, y_bar)
        if backwards:
            self.optimizer.zero_grad()
            loss_l[0].backward()
            self.optimizer.step()
            with open('./trainingData.txt.gz','ab') as f:
                f.write(','.join(str(i) for i in x_copy)+','+str(y)+'\n')
        else:
            with open('./testingData.txt.gz','ab') as f:
                f.write(','.join(str(i) for i in x_copy)+','+str(y)+'\n')
        return [l.data.item() for l in loss_l]

    def run_epoch(self, data_loader, test=False, ryu_test=False):
        data_loader=self.data_loader
        #data_loader.switch_train((not test) and (self.auto))
        self.sumPrint.start_epoch(self.epoch, len(data_loader))
        for j, (data, target) in enumerate(data_loader):
            self.sumPrint.start_iter(j)
            res = self._iter(data, target, self.sumPrint, backwards=not test)
            self.sumPrint.end_iter(j, res)
        
        rets = self.sumPrint.end_epoch()
        self.ctr+=1
        #print(data_loader.dataset.list_of_training_files)
        #exit(1)
        if not test:     
            data_loader.switch_train(test)
            self.network.eval()
            self.validSumPrint.start_epoch(self.epoch, len(data_loader))
            for j, (data, target) in enumerate(data_loader):
                self.validSumPrint.start_iter(j)
                res = self._iter(data, target, self.validSumPrint, backwards=False)
                self.validSumPrint.end_iter(j, res,weights=[1.0,1.0,res[1]/100.0,(100.0-res[1])/100])
            self.network.train()

            val_rets = self.validSumPrint.end_epoch()
        else:
            val_rets = None

        return rets, val_rets


    def ryu_testing(self,val=False,test=True):
        #print("in ryu_testing")
        #data_loader.switch_train((not test) and (self.auto))
        #print(data_loader.dataset.train)
        #exit(1)
        self.sumPrint.start_epoch(self.epoch, len(self.data_loader))
        for j, (data, target) in enumerate(self.data_loader):
            
            self.sumPrint.start_iter(j)
            res = 0
            if val == False:
                #print("calling iter")
                res = self._iter(data, target, self.sumPrint, backwards=False)
            self.sumPrint.end_iter(j, res)
        
        rets = self.sumPrint.end_epoch()
        val_rets = None
        return rets, val_rets

    def test(self, load=True):
        #testing
        print(bcolors.OKBLUE+'*******TESTING********'+bcolors.ENDC)
        data_loader = PFPSampler(self.args, train=False)
        #load checkpoint
        if load:
            self.load()
        #set no gradients
        self.network.eval()
        #run epoch
        
        rets, _ = self.run_epoch(data_loader, True)
        rets = [self.args.run_name] + rets #run name
        return rets

    def ryu_test_procedure(self, load = True):
        print(bcolors.OKBLUE+'*******TESTING********'+bcolors.ENDC)
        self.load()
        self.network.eval()
        while(self.num_norm < self.norm_limit or self.num_ana < self.ana_limit):
            rets, _ = self.ryu_testing(False,not self.data_loader.dataset.train)
            print(self.num_norm, " || ", self.num_ana)

        print(len(self.norm_error))
        print(len(self.ana_error))

        #fit gaussian curve
        if self.fit_curve == True:
            p0 = [1., -1., 1., 1., -1., 1.]
            bin_centres = (min(self.norm_error) + max(self.norm_error))/2
            coeff, var_matrix = curve_fit(gaussx, bin_centres, self.norm_error, p0=p0)
            hist_fit = gauss

 
        rets = [self.args.run_name] + rets #run name
        g_min = min(min(self.ana_error),min(self.norm_error))
        g_max = max(max(self.ana_error),max(self.norm_error))
        bins = np.linspace(g_min,g_max,100)
        plt.hist(self.norm_error,bins,alpha=0.5,label="Norm")
        plt.hist(self.ana_error,bins,alpha=0.5,label="Ano")
        plt.legend(loc='upper right')
        plt.savefig((str)(self.args.run_name)+"-ANO vs NORM errors.png")
        return rets

    def gaussx(x, *p):
        A1, mu1, sigma1= p
        return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))

    def train(self):
        if self.args.load_checkpoint or self.args.resume or (self.args.checkpoint is not None):
            self.load(self.args.resume)
        data_loader = self.data_loader

        print(bcolors.OKBLUE+'*******TRAINING********'+bcolors.ENDC)
        self.network.train()
        while self.epoch < self.args.epochs:
            _, val_ret = self.run_epoch(data_loader, False)
            self.epoch += 1
            if self.epoch % self.args.checkpoint_every == 0:
                print("Saving...")
                self.save()
        self.save()

    def ryu_data_gen(self):
        if self.args.load_checkpoint or self.args.resume or (self.args.checkpoint is not None):
            self.load(self.args.resume)
        data_loader = PFPSampler(self.args,train=True)
        #print(bcolors.OKBLUE+'*******TRAINING*******'+bccolors.ENDC)
        while self.num_norm < self.norm_limit and self.num_ana < self.ana_limit:
            _, val_ret = self.ryu_run_epoch(data_loader, False)
        #print('Normal: ', self.num_norm)
        #print("Ana: ", self.num_ana)
