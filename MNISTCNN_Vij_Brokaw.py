### Aaraj Vij and Jeannine Brokaw
### Data Mining Assignment #3
### CNN Model for MNIST Data Set

"""utils.py"""

import argparse
import subprocess

'''
fun: convert string to boolean: True and False
'''
def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    


"""model.py"""

"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import json
import torch.optim as optim
import torch
import os.path
import argparse
import h5py
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    """docstring for ClassName"""
    
    def __init__(self, args):
        super(CNNModel, self).__init__()
        ##-----------------------------------------------------------
        ## define the model architecture here
        ## MNIST image input size batch * 28 * 28 (one input channel)
        ##-----------------------------------------------------------
        
        ## define CNN layers below
        ##nn.conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = nn.Sequential(nn.Conv2d(1, 14, args.k_size, args.stride, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.BatchNorm2d(14),
                                    nn.Conv2d(14, 10, args.k_size, args.stride, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.4),
                                    nn.BatchNorm2d(10),
                                    nn.Conv2d(10, 6, args.k_size, args.stride, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.4),
                                    nn.BatchNorm2d(6),
                                    nn.MaxPool2d(args.pooling_size,args.pooling_size),
                                )
        
        

        ##------------------------------------------------
        ## write code to define fully connected layer below
        ##------------------------------------------------
        in_size = 864
        out_size = 10
        self.fc = nn.Linear(in_size, out_size)
        

    '''feed features to the model'''
    def forward(self, x):  #default
        
        ##---------------------------------------------------------
        ## write code to feed input features to the CNN models defined above
        ##---------------------------------------------------------
        x_out = self.conv(x)

        ## write flatten tensor code below (it is done)
        x = torch.flatten(x_out,1) # x_out is output of last layer
        
        ## ---------------------------------------------------
        ## write fully connected layer (Linear layer) below
        ## ---------------------------------------------------
        result = self.fc(x)
        
        return result
        
        
"""main.py"""


"""
Fun: CNN for MNIST classification
"""


import numpy as np
import time
import h5py
import argparse
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# from util import _create_batch
import json
import torchvision
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from model import CNNModel
# from utils import str2bool


## input hyper-paras
parser = argparse.ArgumentParser(description = "nueral networks")
parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=25, help="num of epoches")

parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-learning_rate", dest ="learning_rate", type=float, default=0.001, help = "learning rate")
parser.add_argument("-decay", dest ="decay", type=float, default=0.5, help = "learning rate")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
parser.add_argument("-dropout", dest ="dropout", type=float, default=0.4, help = "dropout prob")
parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
parser.add_argument("-load_checkpoint", dest="load_checkpoint", type=str2bool, default=True, help="true of false")

parser.add_argument("-activation", dest="activation", type=str, default='relu', help="activation function")
# parser.add_argument("-MC", dest='MC', type=int, default=10, help="number of monte carlo")
parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
parser.add_argument("-k_size", dest='k_size', type=int, default=4, help="size of filter")
parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")


args = parser.parse_args()
    

def _load_data(DATA_PATH, batch_size):
    '''Data loader'''

    print("data_path: ", DATA_PATH)
    train_trans = transforms.Compose([transforms.RandomRotation(args.rotation),transforms.RandomHorizontalFlip(),\
                                transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True,train=True, transform=train_trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    
    ## for testing
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=False, transform=test_trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, test_loader



def _compute_accuracy(y_pred, y_batch):
    ## please write the code below ##
    return (y_pred==y_batch).sum().item()
    


def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
    lr = learning_rate
    if (epoch > 5):
        lr = 0.001
    if (epoch >= 10):
        lr = 0.0001
    if (epoch > 20):
        lr = 0.00001
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print("learning_rate: ", lr)
    
    
# def _test_model(model):
# 	## you do not have to write it here ##
    
# 	return None
    

def main():

    use_cuda = torch.cuda.is_available() ## if have gpu or cpu 
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(device)
    if use_cuda:
        torch.cuda.manual_seed(72)

    ## initialize hyper-parameters
    num_epoches = args.num_epoches
    decay = args.decay
    learning_rate = args.learning_rate


    ## Load data
    DATA_PATH = "./data/"
    train_loader, test_loader=_load_data(DATA_PATH, args.batch_size)

    ##-------------------------------------------------------
    ## please write the code about model initialization below
    ##-------------------------------------------------------
    model = CNNModel(args) #kernel size, stride

    ## to gpu or cpu
    model.to(device)
    
    ## --------------------------------------------------
    ## please write the LOSS FUNCTION ##
    ## --------------------------------------------------
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)  ## optimizer
    loss_fun = nn.CrossEntropyLoss()   ## cross entropy loss
    
    ##--------------------------------------------
    ## load checkpoint below if you need
    ##--------------------------------------------
    # if args.load_checkpoint == True:
        ## write load checkpoint code below

    
    ##  model training
    print("TRAINING STARTED")
    if args.mode == 'train':
        model = model.train()
        for epoch in range(num_epoches): #10-50
            print("\nEPOCH " + str(epoch))
            ## learning rate
            adjust_learning_rate(learning_rate, optimizer, epoch, decay)

            for batch_id, (x_batch,y_labels) in enumerate(train_loader):
                if(batch_id%50 ==0):
                  print("Batch " + str(batch_id))
                x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

                ## feed input data x into model
                output_y = model(x_batch)
                # output_y = model(torch.squeeze(x_batch))
                
                ##---------------------------------------------------
                ## write loss function below, refer to tutorial slides
                ##----------------------------------------------------
                loss = loss_fun(output_y, y_labels)
                

                ##----------------------------------------
                ## write back propagation below
                ##----------------------------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ##------------------------------------------------------
                ## get the predict result and then compute accuracy below
                ## please refer to defined _compute_accuracy() above
                ##------------------------------------------------------
                _, y_pred = torch.max(output_y.data, 1)
                # print(_compute_accuracy(y_pred, y_labels))
                
                
                
                ##----------------------------------------------------------
                ## loss.item() or use tensorboard to monitor the loss blow
                ## if use loss.item(), you may use log txt files to save loss
                ##----------------------------------------------------------
                if(batch_id%50 == 0):
                  print("Loss is", loss.item())
                  model.eval()
                  with torch.no_grad():
                    for batch_id, (x_batch,y_labels) in enumerate(test_loader):
                      x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
                      ##------------------------------------
                      ## write the predict result below
                      ##------------------------------------
                      output_y = model(x_batch)

                      ##--------------------------------------------------
                      ## write code for computing the accuracy below
                      ## please refer to defined _compute_accuracy() above
                      ##---------------------------------------------------
                      _, y_pred = torch.max(output_y.data, 1)
                      print("Test accuracy is " , _compute_accuracy(y_pred, y_labels))
                      model = model.train()
                      break;

            ## -------------------------------------------------------------------
            ## save checkpoint below (optional), every "epoch" save one checkpoint
            ## -------------------------------------------------------------------
            
            print("\nEpoch ", epoch, " results:" )
            model.eval()
            with torch.no_grad():
              accuracy = 0;
              for batch_id, (x_batch,y_labels) in enumerate(test_loader):
                x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
                
                output_y = model(x_batch)

                
                _, y_pred = torch.max(output_y.data, 1)
                accuracy += _compute_accuracy(y_pred, y_labels)
              
              model = model.train()
              print("Test accuracy is " , accuracy/batch_id)

            
                

    ##------------------------------------
    ##    model testing code below
    ##------------------------------------
    print("TEST ACCURACY")
    model.eval()
    with torch.no_grad():
        for batch_id, (x_batch,y_labels) in enumerate(test_loader):
            x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
            ##------------------------------------
            ## write the predict result below
            ##------------------------------------
            output_y = model(x_batch)

            ##--------------------------------------------------
            ## write code for computing the accuracy below
            ## please refer to defined _compute_accuracy() above
            ##---------------------------------------------------
            _, y_pred = torch.max(output_y.data, 1)
            print(_compute_accuracy(y_pred, y_labels))
        

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
    