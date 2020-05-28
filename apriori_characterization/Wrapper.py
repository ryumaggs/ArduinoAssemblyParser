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
import scipy.optimize
import curve_fit
from Logger import SummaryPrint
from misc import bcolors
import misc
import matplotlib.pyplot as plt
from DataLoaders import PFPSampler
from scipy.stats import shapiro
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

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
        self.norm_limit = 200
        self.ana_limit = 200
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
        print('data loader: ', data_loader)
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
        if backwards == False: #and self.args.ryu_testing == True:
            batch = 0
            batch_test = []
            while batch < 512:
                label = y[batch].item()

                if label == 2 and self.num_norm > self.norm_limit:
                    return [l.data.item() for l in loss_l]
                if label == 1 and self.num_ana > self.ana_limit:
                    return [l.data.item() for l in loss_l]
                truth = x[batch,:,:].cpu().numpy()
                batch_test.append(truth)

                guess = y_bar[batch,:,:].detach().cpu().numpy()
                l2 = np.linalg.norm(truth-guess)

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

    # def _ryu_iter(self, x, y, sumPrint, backwards=True):
    #     x_copy = torch.Tensor.numpy(x)
    #     x_copy = x_copy.tolist()
    #     #print("---------------", len(x_copy))
    #
    #     x,y = x.to(self.device), y.to(self.device)
    #     y_bar = self.network(x)
    #     loss_l = self.network.loss(x, y, y_bar)
    #     if backwards:
    #         self.optimizer.zero_grad()
    #         loss_l[0].backward()
    #         self.optimizer.step()
    #         with open('./trainingData.txt.gz','ab') as f:
    #             f.write(','.join(str(i) for i in x_copy)+','+str(y)+'\n')
    #     else:
    #         with open('./testingData.txt.gz','ab') as f:
    #             f.write(','.join(str(i) for i in x_copy)+','+str(y)+'\n')
    #     return [l.data.item() for l in loss_l]
    def gather_recon(self):
        data_loader=self.data_loader
        data_loader.switch_train(False)

        r = []
        enum_data = enumerate(data_loader, 0)
        for i, data in enum_data:
            print(i)
            # get the inputs; data is a list of [inputs, labels]
            input, label = data
            # load these tensors into gpu memory
            input = input.cuda()
            # check if the inputs are cpu or gpu tensor
            output = self.network(input)
            r_error,test_perc,anom_loss,norm_loss = self.network.loss(input, label, output)
            inputs.append(input)
            r.append(r_error)

        return r

    def fit_recon(self, r):
        mean, var = norm.stats(r)
        range = [mean + var, mean - var]
        anom = []
        for error in r:
            if error > range[0] or error < range[1]:
                anom.append(True)
            else:
                anom.append(False)

        return anom

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

    def metrics(y_actual, y_hat):

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(labels)):
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return precision, recall

    # def prc(precision, recall)
    def recon_errors(self, data_loader):
        # Gather recon errors on data
        r = []
        labels = []
        for i, data in enumerate(data_loader, 0):
            input, label = data
            labels.append(label)
            # load these tensors into gpu memory
            input = input.cuda()
            # check if the inputs are cpu or gpu tensor
            output = self.network(input)
            r_error,test_perc,anom_loss,norm_loss = self.network.loss(input, label, output)
            # inputs.append(input)
            r_item = r_error.item()
            # print(r_item)

            r.append(r_item)

        return r, labels

    def fit_recon_to_norm(self, r):
        # Plot training recon error distribution fit to gaussian
        mean = np.mean(r)
        std = np.std(r)

        range = [mean + std, mean - std]
        anom = []

        # Plot the histogram.
        plt.hist(r, bins=25, density=True, alpha=0.6, color='g')

        mu, std_norm = norm.fit(r)

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x =  np.sort(r)#np.linspace(xmin, xmax, r_len)
        p = norm.pdf(x, mu, std_norm)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mean, std)
        plt.title(title)

        plt.show()
        currentDirectory = os.getcwd()
        print(currentDirectory)
        plt.savefig(currentDirectory + '/fit.png')
        plt.close()

        return mean, std

    def test(self, load=True):
        #testing
        print(bcolors.OKBLUE+'*******TESTING********'+bcolors.ENDC)

        data_loader = PFPSampler(self.args, train=True)
        data_loader_test = PFPSampler(self.args, train=False)
        #load checkpoint
        if load:
            self.load()
        #set no gradients
        print('setting no grad')
        self.network.cuda()
        self.network.eval()
        #run epoch
        self.data_loader.switch_train(True)

        # r = gather_recon()
        # while(self.num_norm < self.norm_limit or self.num_ana < self.ana_limit):
        #     rets, _ = self.ryu_testing(False, not self.data_loader.dataset.train)
        #     rets, _ = self.run_epoch(self.data_loader, True)
        #     print(self.num_norm, " || ", self.num_ana)

        r, training_labels = self.recon_errors(data_loader)
        training_labels = torch.cat(training_labels, dim=0)
        training_labels = torch.flatten(training_labels)

        mean, std = self.fit_recon_to_norm(r)

        # Gather testing recon errors
        self.data_loader.switch_train(False)
        r_test, testing_labels = self.recon_errors(data_loader_test)


        # Visuals
        testing_labels = torch.cat(testing_labels, dim=0)
        testing_labels = torch.flatten(testing_labels)

        test_label_list = testing_labels.tolist()
        self.std3(mean, std, r_test, test_label_list)
        self.chevy(mean, std, r_test, test_label_list)
        print("----------------")
        rets, _ = self.run_epoch(self.data_loader, True)
        rets = [self.args.run_name] + rets #run name
        print("rets", rets)
        return rets

    def prc(self, precision, recall, type):
        # calculate precision-recall AUC
        auc = auc(recall, precision)
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(label="{}, AUC={:.3f}".format(auc))
        # show the legend
        plt.legend()
        # show the plot
        currentDirectory = os.getcwd()

        if type == 'C':
            plt.savefig(currentDirectory + '/Chevy-PRC.png')
        else:
            plt.savefig(currentDirectory + '/STD3-PRC.png')


    def std3(self, mean, std, r, labels):
        # STD*3
        pred_labels = []
        cut_off = std * 3
        lower, upper = mean - cut_off, mean + cut_off
        outliers_std3 = [err for err in r if err < lower or err > upper]
        for err in r:
            if err in outliers_std3:
                pred_labels.append(True) # anomaly
            else:
                pred_labels.append(False) #normal

        # precision, recall = metrics(r_test, pred_r)
        precision, recall, thresholds = precision_recall_curve(labels, pred_labels)
        self.prc(precision, recall,'S')

    def chevy(self, mean, std, r, labels):
        # Chevyshev http://kyrcha.info/2019/11/26/data-outlier-detection-using-the-chebyshev-theorem-paper-review-and-online-adaptation

        # Stage 1

        p1 = 0.1 # probability of expected outlier
        k = 4.472 # 1 / sqrt(p1)
        lower = mean - (k * std)
        upper = mean + (k * std)

        # Data that are more extreme than the ODVs of stage-1 are removed from the data for the second phase of the algorithm.
        advance = [err for err in r if err > lower and err < upper]
        print("advancing")
        print(advance)
        stage_1_behind = [err for err in r if err < lower or err > upper]

        # Stage 2
        p2 = 0.01
        k2 = 10
        mean_trunc= np.mean(advance)
        std_trunc = np.std(advance)

        lower = mean_trunc - (k2 * std_trunc)
        upper = mean_trunc + (k2 * std_trunc)

        print("LOWER: ", lower)
        print("UPPER: ", upper)
        print("R: ", r)
        outliers = [err for err in r if err < lower or err > upper]
        stage_2_behind = [err for err in r if err > lower or err < upper]

        pred_labels = []

        for err in r:
            if err in outliers:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        precision, recall, thresholds = precision_recall_curve(labels, pred_labels)
        self.prc(precision, recall,'C')


        # return outliers

    def roc(labels, r_error):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr, tpr, _ = metrics.roc_curve(labels, r_error, pos_label=1)
        print("threshold: ", _)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Reconstruction Error ROC')
        plt.legend(loc="lower right")
        currentDirectory = os.getcwd()
        plt.savefig(currentDirectory + '/ROC.png')

    def ryu_test_procedure(self, load = True):
        print(bcolors.OKBLUE+'*******TESTING********'+bcolors.ENDC)
        self.load()
        self.network.eval()
        while(self.num_norm < self.norm_limit or self.num_ana < self.ana_limit):
            rets, _ = self.ryu_testing(False,not self.data_loader.dataset.train)
            print(self.num_norm, " || ", self.num_ana)

        rets = [self.args.run_name] + rets #run name
        return rets


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

    # def train(self):
    #     x = 100
    #     for epoch in range(x):  # loop over the dataset multiple times
    #
    #         running_loss = 0.0
    #         for i, data in enumerate(PFPSampler, 0):
    #
    #             # get the inputs; data is a list of [inputs, labels]
    #             inputs, labels = data
    #
    #             # zero the parameter gradients
    #             self.optimizer.zero_grad()
    #
    #             # forward + backward + optimize
    #             outputs = self.network(inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()
    #
    #             # print statistics
    #             running_loss += loss.item()
    #             if i % 2000 == 1999:    # print every 2000 mini-batches
    #                 print('[%d, %5d] loss: %.3f' %
    #                       (epoch + 1, i + 1, running_loss / 2000))
    #                 running_loss = 0.0

print('Finished Training')

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

    # def ryu_data_gen(self):
    #     if self.args.load_checkpoint or self.args.resume or (self.args.checkpoint is not None):
    #         self.load(self.args.resume)
    #     data_loader = PFPSampler(self.args,train=True)
    #     #print(bcolors.OKBLUE+'*******TRAINING*******'+bccolors.ENDC)
    #     while self.num_norm < self.norm_limit and self.num_ana < self.ana_limit:
    #         _, val_ret = self.ryu_run_epoch(data_loader, False)
    #     #print('Normal: ', self.num_norm)
    #     #print("Ana: ", self.num_ana)
