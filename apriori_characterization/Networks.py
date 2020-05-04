import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

class AutoConvNetwork(nn.Module):
    @staticmethod
    def add_args(parser):

        parser.add_argument('--enc_channels', metavar='N', type=int, nargs='+', default=[16,16,32,32])
        parser.add_argument('--hidden_size', metavar='N', type=int, nargs='+',default=[140,90,60,90,140])
        parser.add_argument('--dec_channels', metavar='N', type=int, nargs='+', default=[20,16,12,9,6,4,3,1])
        parser.add_argument('--stride', metavar='N', type=int, default=2)
        parser.add_argument('--filter_size', metavar='N', type=int, default=4)
        parser.add_argument('--dec_reg',type=str,default="None",help="None,L1,L2...")
        parser.add_argument('--mid_dropout',type=float,default=0,help="middle layer dropout value from 0 to 1")
        parser.add_argument('--loss_type',type=str,default="L2",help="L1,L2...")
        
    def __init__(self, args, x_size):
        super(AutoConvNetwork, self).__init__()
        def conv2d_out_size(Hin, layer):
            return int((Hin+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1) / (layer.stride[0])+1)
        def convTranspose1d_out_size(Hin, stride, padding, kernel_size, output_padding):
            return int((Hin-1)*stride-2*padding+kernel_size+output_padding)
        
        cur_channel = 1
        self.batch_size = args.batch_size
        cur_height = x_size[2]


        #encode layers
        enc_layers = []
        enc_height = [cur_height]
        for i, ec in enumerate(args.enc_channels):
            enc_layers.append(nn.Conv1d(cur_channel, ec, kernel_size=args.filter_size, stride=args.stride, padding=0))
            cur_height = conv2d_out_size(cur_height, enc_layers[-1])
            enc_height.append(cur_height)
            enc_layers.append(nn.ReLU())
            #enc_layers.append(nn.BatchNorm1d(ec))
            cur_channel = ec
        
        self.enc_layers = nn.Sequential(*enc_layers)
        print(self.enc_layers)
        self.im_size = (cur_channel, cur_height)
        self.flat_size = int(cur_channel * cur_height)
        temp=cur_channel
        hidden_layers = []
        cur_channel=self.flat_size
        print(cur_channel,cur_channel,cur_channel)
        h_sizes=args.hidden_size+[self.flat_size]
        for i,hl in enumerate(h_sizes):
            hidden_layers.append(nn.Linear(cur_channel,hl))
            cur_channel=hl
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.BatchNorm1d(hl))

        self.hidden_layers = nn.Sequential(*hidden_layers)
        print(self.hidden_layers)
        cur_channel=temp
        dec_layers = []
        for i, dc in enumerate(args.dec_channels):
            out_height = convTranspose1d_out_size(cur_height, args.stride, 0, args.filter_size, 0)
            output_padding = enc_height[len(args.enc_channels)-i-1]-out_height
            cur_height = out_height + output_padding
            dec_layers.append(nn.ConvTranspose1d(cur_channel, dc, kernel_size=args.filter_size, stride=args.stride, padding=0, output_padding=output_padding))
            if i != len(args.dec_channels)-1:
                dec_layers.append(nn.ReLU())
            cur_channel = dc
            dec_layers.append(nn.BatchNorm1d(dc))
        self.dec_layers = nn.Sequential(*dec_layers)
        print(self.dec_layers)
        self.recon_loss = nn.MSELoss()


    def forward(self, x):
        
        #encode layers
        enc = self.enc_layers(x)
        #hidden layers
        enc_flat = enc.view((-1, self.flat_size))
        hid_flat = self.hidden_layers(enc_flat)
        hid = hid_flat.view(enc.shape)
        #decode layers
        dec = self.dec_layers(hid)
        
        return dec

    def loss(self, x, y, y_bar):
        recon_loss = self.recon_loss(y_bar, x)
        test_perc = 100.0*torch.sum(y)/y.size()[0]
        yb2=y_bar[(y==1)[:,0],:,:]
        x2=x[(y==1)[:,0],:,:]
        anom_loss = self.recon_loss(yb2,x2)
        yb2=y_bar[(y==0)[:,0],:,:]
        x2=x[(y==0)[:,0],:,:]
        norm_loss = self.recon_loss(yb2,x2)
        if(anom_loss!=anom_loss):
            anom_loss=torch.Tensor([0.0])
        return [recon_loss,test_perc,anom_loss,norm_loss]
        
    def loss_names(self):
        return ['rec_loss','test_perc','anom_loss','norm_loss']

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class LatentAutoConvNetwork(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--feature_channels', metavar='N', type=int, nargs='+', default=[2,3,4,6,9,12,16,20])
        parser.add_argument('--latent_size', metavar='N', type=int, default=[1500,1000])
        parser.add_argument('--enc_channels', metavar='N', type=int, nargs='+', default=[2,3,4,5])
        parser.add_argument('--hidden_size', metavar='N', type=int, default=[25,20,25])
        parser.add_argument('--dec_channels', metavar='N', type=int, nargs='+', default=[4,3,2,1])
        parser.add_argument('--stride', metavar='N', type=int, default=2)
        parser.add_argument('--filter_size', metavar='N', type=int, default=4)
        parser.add_argument('--dec_reg',type=str,default="None",help="None,L1,L2...")
        parser.add_argument('--mid_dropout',type=float,default=0,help="middle layer dropout value from 0 to 1")
        parser.add_argument('--loss_type',type=str,default="L2",help="L1,L2...")
        
    def __init__(self, args, x_size):
        super(LatentAutoConvNetwork, self).__init__()
        def conv2d_out_size(Hin, layer):
            return int((Hin+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1) / (layer.stride[0])+1)
        def convTranspose1d_out_size(Hin, stride, padding, kernel_size, output_padding):
            return int((Hin-1)*stride-2*padding+kernel_size+output_padding)
        
        cur_channel = 1
        self.batch_size = args.batch_size
        cur_height = x_size[2] #The window Size

        #print(cur_height)
        #exit(1)
        #encode layers
        feature_enc_layers = []
        feature_enc_height = [cur_height]
        for i, ec in enumerate(args.feature_channels):
            feature_enc_layers.append(nn.Conv1d(cur_channel, ec, kernel_size=args.filter_size, stride=args.stride, padding=0))
            cur_height = conv2d_out_size(cur_height, feature_enc_layers[-1])
            feature_enc_height.append(cur_height)
            feature_enc_layers.append(nn.ReLU())
            feature_enc_layers.append(nn.BatchNorm1d(ec))
            cur_channel = ec




        self.feature_enc_layer = nn.Sequential(*feature_enc_layers)
        



        print(self.feature_enc_layer)
        #print(feature_enc_height)
        cur_channel = args.feature_channels[-1]
        cur_height = feature_enc_height[-1]
        #print(cur_channel, cur_height)
        #exit(1)
        self.im_size = (cur_channel, cur_height)
        self.feat_flat_size = int(cur_channel * cur_height)
        #print(self.flat_size)
        #exit(1)
        hidden_layers = []
        cur_channel=self.feat_flat_size
        print(cur_channel,cur_channel,cur_channel)
        h_sizes=args.latent_size
        print(h_sizes)
        for i,hl in enumerate(h_sizes):
            hidden_layers.append(nn.Linear(cur_channel,hl))
            cur_channel=hl
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.BatchNorm1d(hl))



        self.feature_hidden_layers = nn.Sequential(*hidden_layers)




        cur_channel = 1
        cur_height = h_sizes[len(h_sizes)-1]
        print(self.feature_hidden_layers)
        #exit(1)
        enc_layers = []
        enc_height = []
        for i, ec in enumerate(args.enc_channels):
            enc_layers.append(nn.Conv1d(cur_channel, ec, kernel_size=args.filter_size, stride=args.stride, padding=0))
            cur_height = conv2d_out_size(cur_height, enc_layers[-1])
            print(cur_height)
            enc_height.append(cur_height)
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.BatchNorm1d(ec))
            cur_channel = ec
        #print(enc_layers)


        self.enc_layers = nn.Sequential(*enc_layers)


        self.im_size = (cur_channel, cur_height)
        print(self.enc_layers)
        #print(cur_channel)
        #print(cur_height)
        #exit(1)
        cur_channel = args.enc_channels[-1]
        cur_height = enc_height[-1]
        self.flat_size = int(cur_channel * cur_height)
        hidden_layers = []
        cur_channel=self.flat_size
        print(cur_channel,cur_channel,cur_channel)
        h_sizes=args.hidden_size+[self.flat_size]
        print(h_sizes)
        for i,hl in enumerate(h_sizes):
            hidden_layers.append(nn.Linear(cur_channel,hl))
            cur_channel=hl
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.BatchNorm1d(hl))



        self.hidden_layers = nn.Sequential(*hidden_layers)


        print(self.hidden_layers)
        #exit(1)
        cur_channel=args.enc_channels[-1]
        cur_height=enc_height[-1]
        print(cur_channel)
        print(cur_height)
        #exit(1)
        dec_layers = []
        print(enc_height)
        dec_height = [args.latent_size[-1]] + enc_height[:-1]
        dec_height.reverse()
        print(dec_height)
        #exit(1)
        for i, dc in enumerate(args.dec_channels):
            out_height = convTranspose1d_out_size(cur_height, args.stride, 0, args.filter_size, 0)
            print(enc_height[len(args.enc_channels)-i-2])
            print('*',out_height)
            output_padding = dec_height[i]-out_height
            print(output_padding)
            cur_height = out_height + output_padding
            dec_layers.append(nn.ConvTranspose1d(cur_channel, dc, kernel_size=args.filter_size, stride=args.stride, padding=0, output_padding=output_padding))
            if i != len(args.dec_channels)-1:
                dec_layers.append(nn.ReLU())
            cur_channel = dc
            dec_layers.append(nn.BatchNorm1d(dc))
        print("--",cur_height)
        self.dec_layers = nn.Sequential(*dec_layers)

        #exit(1)
        print(self.dec_layers)
        #exit(1)
        self.recon_loss = nn.MSELoss()
   
    def forward(self, x):
        #feature_layer
        feat = self.feature_enc_layer(x)
        feat_flat = feat.view((-1,self.feat_flat_size))
        feat_hid = self.feature_hidden_layers(feat_flat)
        #print(feat_hid.shape)
        feat_hid = feat_hid.view((feat_hid.shape[0],1,feat_hid.shape[1]))
        #print(feat_hid.shape)
        
        #encode layers
        enc = self.enc_layers(feat_hid)
        #exit(1)
        #hidden layers
        enc_flat = enc.view((-1, self.flat_size))
        hid_flat = self.hidden_layers(enc_flat)
        hid = hid_flat.view(enc.shape)
        #decode layers
        dec = self.dec_layers(hid)
        
        return dec

    def loss(self, x, y, y_bar):
        feat = self.feature_enc_layer(x)
        feat_flat = feat.view((-1,self.feat_flat_size))
        feat_hid = self.feature_hidden_layers(feat_flat)
        #print(feat_hid.shape)
        x = feat_hid.view((feat_hid.shape[0],1,feat_hid.shape[1])) 
        #print(y_bar.shape)
        #print(x.shape)
        #exit(1)
        recon_loss = self.recon_loss(y_bar, x)
        test_perc = 100.0*torch.sum(y)/y.size()[0]
        yb2=y_bar[(y==1)[:,0],:,:]
        x2=x[(y==1)[:,0],:,:]
        anom_loss = self.recon_loss(yb2,x2)
        yb2=y_bar[(y==0)[:,0],:,:]
        x2=x[(y==0)[:,0],:,:]
        norm_loss = self.recon_loss(yb2,x2)
        if(anom_loss!=anom_loss):
            anom_loss=torch.Tensor([0.0])
        return [recon_loss,test_perc,anom_loss,norm_loss]
        
    def loss_names(self):
        return ['rec_loss','test_perc','anom_loss','norm_loss']


class ClassConvNetwork(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--conv_layers', metavar='N', type=int, nargs='+', default=[8,8,16,16,32,32])
        parser.add_argument('--fc_layers', metavar='N', type=int, default=[256,64,2])
        parser.add_argument('--stride', metavar='N', type=int, default=3)
        parser.add_argument('--filter_size', metavar='N', type=int, default=4)
        parser.add_argument('--dec_reg',type=str,default="None",help="None,L1,L2...")
        parser.add_argument('--mid_dropout',type=float,default=0,help="middle layer dropout value from 0 to 1")
        parser.add_argument('--loss_type',type=str,default="CE",help="L1,L2...")
        
    def __init__(self, args, x_size):
        super(ClassConvNetwork, self).__init__()
        def conv2d_out_size(Hin, layer):
            return int((Hin+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1) / (layer.stride[0])+1)
        def convTranspose1d_out_size(Hin, stride, padding, kernel_size, output_padding):
            return int((Hin-1)*stride-2*padding+kernel_size+output_padding)
        
        cur_channel = 1
        self.batch_size = args.batch_size
        cur_height = x_size[2]


        #encode layers
        enc_layers = []
        enc_height = [cur_height]
        for i, ec in enumerate(args.conv_layers):
            enc_layers.append(nn.Conv1d(cur_channel, ec, kernel_size=args.filter_size, stride=args.stride, padding=0))
            cur_height = conv2d_out_size(cur_height, enc_layers[-1])
            enc_height.append(cur_height)
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.BatchNorm1d(ec))
            cur_channel = ec
            print(cur_height,ec)
        
        enc_layers.append(Flatten())
        self.im_size = (cur_channel, cur_height)
        self.flat_size = int(cur_channel * cur_height)
        temp=cur_channel
        cur_channel=self.flat_size
        hidden_layers=[]
        print(cur_channel,cur_channel,cur_channel)
        for i,hl in enumerate(args.fc_layers):
            hidden_layers.append(nn.Linear(cur_channel,hl))
            cur_channel=hl
            #enc_layers.append(nn.BatchNorm1d(hl))
            hidden_layers.append(nn.ReLU())
        self.recon_loss = nn.CrossEntropyLoss()
        self.layers=nn.Sequential(*(enc_layers+hidden_layers))
        print(enc_layers+hidden_layers)

    def forward(self, x):
        
        #encode layers
        dec = self.layers(x)
   
        return dec

    def loss(self, x, y, y_bar):
        y=y[:,0]
        recon_loss = self.recon_loss(y_bar, y)
        acc=self.acc(x,y,y_bar)
        return [recon_loss,acc,torch.tensor(0),torch.tensor(0)]
        

    def acc(self, x,y,y_bar):
        predicted = torch.argmax(y_bar, 1)
        acc = torch.mean((predicted == y).float())
        return acc

    def loss_names(self):
        return ['class_loss','acc','0','0']

class LSTM(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--conv_layers', metavar='N', type=int, nargs='+', default=[2,3,4,6,9,12,16,20])
        parser.add_argument('--fc_layers', metavar='N', type=int, default=[48,16,2])
        parser.add_argument('--stride', metavar='N', type=int, default=2)
        parser.add_argument('--filter_size', metavar='N', type=int, default=4)
        parser.add_argument('--dec_reg',type=str,default="None",help="None,L1,L2...")
        parser.add_argument('--mid_dropout',type=float,default=0,help="middle layer dropout value from 0 to 1")
        parser.add_argument('--loss_type',type=str,default="CE",help="L1,L2...")
        parser.add_argument('--input_dim', metavar='N', type=int, nargs='+', default=128)
        parser.add_argument('--hidden_dim', metavar='N', type=int, nargs='+', default=256)
        parser.add_argument('--num_layers', metavar='N', type=int, nargs='+', default=2)
        parser.add_argument('--output_dim', metavar='N', type=int, nargs='+', default=128)

    def __init__(self, args, x_size):
        super(LSTM, self).__init__()
        def conv2d_out_size(Hin, layer):
            return int((Hin+2*layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1) / (layer.stride[0])+1)
        def convTranspose1d_out_size(Hin, stride, padding, kernel_size, output_padding):
            return int((Hin-1)*stride-2*padding+kernel_size+output_padding)

        cur_channel = 1
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        cur_height = x_size[2]

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim,self.output_dim)
        self.recon_loss = nn.MSELoss()

    def init_hidden(self): #not used as of now
        return (torch.zeros(self.num_layers,self.batch_size,self.hidden_dim),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)
