import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import os
import json, fnmatch
import sys
import re
import random
from random import shuffle
import time
#imports for zero_padding test
import matplotlib.pyplot as plt

class PFPSampler2(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--files_per_bucket', type=int, default=8, help='n files per bucket fill')
        parser.add_argument('--datalen', type=int, default=16384, help= 'epoch size')
        parser.add_argument('--window_size', type=int, default=48000, help='window size')
        parser.add_argument('--window_offset', type=float, default=1.0, help='percent of window used for offset')
        parser.add_argument('--train_percent', type=float, default=.7, help='percent of dataset used for training')
        parser.add_argument('--data_dirs', metavar='N', type=str, nargs='+', default=['/scratch/PFPData/LogicModified/traces/'])
        parser.add_argument('--samples_per', type=int, default=64, help='number of data samples taken in each file')
        parser.add_argument('--test_balancer', type=float, default=.09, help='percent of normals kept during testing')

    def __init__(self, args,train):
        print("Initializing Data Loader....")
        self.datalen=args.datalen
        self.bucket = []
        self.window = args.window_size
        self.step = int(args.window_offset*args.window_size)
        self.train=train
        self.class_conv = args.class_conv
        self.train_frac=args.train_percent
        self.data_dirs=[]
        self.data_name = ''.join([i for i in args.data_dirs[0][-15:] if i.isalpha()])
        #print(self.data_name)
        #exit(1)
        self.data_dirs.extend(args.data_dirs)
        self.keep_label=[0]
        if self.class_conv == True:
            self.keep_label = [1,2]
            # print("keeping label 1 & 2")
        print("keeping labels: ", self.keep_label)
        self.list_of_files = fnmatch.filter(os.listdir(self.data_dirs[0]), "*.meta")
        self.list_of_training_files = []
        self.used_list_of_training_files = []
        self.list_of_testing_files = []
        self.used_list_of_testing_files = []
        self.bucket_labels = []
        #checking for scale of data
        self.normal_mean = 0
        self.num_normal = 0
        self.anomalous_mean = 0
        self.num_anomalous = 0
        self.normal_histo = []
        self.anomalous_histo = []
        #checking for zero-padding in data
        #self.zero_padding = []
        print("Adding initial batch of files...")
        for f in self.list_of_files:
            file_name = self.data_dirs[0] + f
            with open(file_name, mode='r') as metafile:
                meta_information = json.load(metafile)
                label = meta_information['core:global']['PFP:label']
                if label in self.keep_label and train == True:
                    train_or_valid = random.randint(0,4)
                    if label not in self.keep_label:
                        print("Something is wrong with data loader")
                        exit(1)
                    if train_or_valid < 3:
                        self.list_of_training_files.append(f)
                    else:
                        self.list_of_testing_files.append(f)
                if label in self.keep_label and train == False:
                    if label not in self.keep_label:
                        print("something wrong with data loader...")
                        exit(1)
                    self.list_of_testing_files.append(f)
                    self.bucket_labels.append(label)
        if train == True:
            self.num_files=len(self.list_of_training_files)
            shuffler=np.random.permutation(self.num_files)
            self.list_of_training_files=[self.list_of_training_files[i] for i in shuffler]
            print("training files: ", self.list_of_training_files)
        else:
            self.num_files = len(self.list_of_testing_files)
            shuffler=np.random.permutation(self.num_files)
            self.list_of_testing_files=[self.list_of_testing_files[i] for i in shuffler]


        print("Done")
        self.files_per_bucket = args.files_per_bucket
        self.samples_per =args.samples_per
        self.rebalance=args.test_balancer
        self.ct=0
    
    def __getitem__(self, index):
        if len(self.bucket)<=0:
            self._fill_bucket()
        #checking for data scale via histogram

        print("anom histo: ", self.anomalous_histo)
        print("normal histo: ", self.normal_histo)


        g_min = min(min(self.anomalous_histo), min(self.normal_histo))
        g_max = max(max(self.anomalous_histo),max(self.normal_histo))
        bins = np.linspace(g_min,g_max,100)
        plt.hist(self.normal_histo,bins,alpha=0.5,label="Norm")
        plt.hist(self.anomalous_histo,bins,alpha=0.5,label="Ano")
        plt.legend(loc='upper right')
        plt.savefig((str)(self.data_name)+"-Scale_check.png")
        exit(1)
        return torch.from_numpy(np.asarray(self.bucket.pop())),torch.tensor([self.bucket_labels.pop()])

    def __len__(self):
        return self.datalen
    
    def switch_train(self,newtrain):
        self.train=newtrain

    def _fill_bucket(self):
        if self.train:
            for i in range(self.files_per_bucket):
                if len(self.list_of_training_files) == 0:
                    self.list_of_training_files = self.used_list_of_training_files
                    self.used_list_of_training_files = []
                meta_file=random.choice(self.list_of_training_files)
                self.list_of_training_files.remove(meta_file)
                self.used_list_of_testing_files.append(meta_file)
                good_file=self._handle_file(meta_file)

        else:
            for i in range(self.files_per_bucket):
                if len(self.list_of_testing_files) == 0:
                    self.list_of_testing_files = self.used_list_of_testing_files
                    self.used_list_of_testing_files = []
                meta_file=random.choice(self.list_of_testing_files)
                self.list_of_testing_files.remove(meta_file)
                self.used_list_of_testing_files.append(meta_file)
                good_file=self._handle_file(meta_file)
            #check for zero-padding: 
            #zero_pad = np.array(self.zero_padding)
            #x_min = 0
            #x_max = len(self.zero_padding[0])
            #for i in range(len(self.zero_padding)):
            #    plt.plot(self.zero_padding[i])
            #plt.savefig("zero_padding_test.png")
            #exit(1)

        #checking for data scale
        #self.normal_mean = self.normal_mean / self.num_normal
        #self.anomalous_mean = self.anomalous_mean / self.num_anomalous
        #print("Average mean of normal data: ", self.normal_mean)
        #print("Average mean of ano data: ", self.anomalous_mean)
        #exit(1)    
       
            

    def _handle_file(self,file_name):
        file_name=self.data_dirs[0]+file_name
        data_file=file_name.replace("meta","data")
        data = np.fromfile(data_file, dtype=np.float32)
        with open(file_name, mode='r') as metafile:
            meta_information = json.load(metafile)
            data_channels,length =self._channel_trigger_data(meta_information, data)
            DATA=self._chop_windows(data_channels,length)
            self.bucket.extend(DATA)    
            label = meta_information['core:global']['PFP:label']
            self.bucket_labels.extend([label]*len(DATA))
            #check for scale of each class:
            for row in DATA:
                temp_mean = np.mean(row)
                if label == 2:
                    #self.normal_mean += temp_mean
                    #self.num_normal += 1
                    #print("Label", label, " mean of ", temp_mean)
                    self.normal_histo.append(temp_mean)
                elif label == 1:
                    #self.anomalous_mean += temp_mean
                    #self.num_anomalous += 1
                    #print("Label", label, " mean of ", temp_mean)
                    self.anomalous_histo.append(temp_mean)
        return True

    def _chop_windows(self,data_channels,length):
        rnd_ints=[]
        window=self.window
        for i in range(self.samples_per):
            rnd_ints.append(random.randint(0,length-window-1))
        DATA=[]
        dc=np.asarray(data_channels)
        dc = dc[0,:]
        dc = np.reshape(dc,(1,dc.shape[0]))
        #checking for zero-padding
        i = dc.shape[1]-10
        zero_counter = 0
        #while(i < dc.shape[1]):
        #    if dc[0][i] == 0:
        #        zero_counter += 1
        #    i+=1

        #if zero_counter > 8:
        #    self.zero_padding.append(dc[0][1000:])

        for i in rnd_ints:
            DATA.append(100.0*dc[:,i:i+window])
        return DATA


    def _channel_trigger_data(self,meta_information, data,plot=False):
        num_channels=0
        data_channels=[]
        for channel in meta_information['core:capture']:
            num_channels+=1
            sample_start = channel['core:sample_start']
            sample_length = channel['PFP:length']
            trigger_sample = channel['PFP:trigger_sample']
            data_channels.append(data[sample_start+trigger_sample:sample_start+sample_length+trigger_sample])
            length=len(data[sample_start+trigger_sample:sample_start+sample_length])
        return data_channels,length


class PFPSampler(object):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--add_later',metavar='l',type=float, help='Time',default=1e-2)
    def __init__(self, args, train):
        self.args = args
        self.dataset=PFPSampler2(self.args, train)
        

        self.data_loader = torch.utils.data.DataLoader(self.dataset,batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=4)
        #self.valid_data_loader=torch.utils.data.DataLoader(dataset2,batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=2)
        #self.valid_iterator = iter(self.valid_data_loader)

    def switch_train(self,newtrain):
        self.dataset.switch_train(newtrain)
        

    def __iter__(self):
        return iter(self.data_loader)

    def valid_batch(self):
        try:
            self.switch_train(False)
            nxt=next(self.data_loader())
            self.switch_train(True)
            return nxt
        except StopIteration:
            self.valid_iterator = iter(self.data_loader)
            return next(self.valid_iterator)

    def __len__(self):
        return len(self.data_loader)

