import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import os
import numpy as np
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from matplotlib import pyplot as plt


class Part3_BasicCNN(nn.Module):

    def __init__(self):

        super(Part3_BasicCNN, self).__init__()

        ######

        self.conv1 = nn.Conv2d(3,4,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4,8,5)

        self.fc1 = nn.Linear(8*11*11, 100)
        self.fc2 = nn.Linear(100, 10)

        ######

    def forward(self, features):
        x = features
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,8*11*11)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class CNN2(nn.Module):

    # def __init__(self, kernels, neurons, con, input_size, channels):
    def __init__(self):

        # self.neurons = neurons
        # self.input_size = input_size
        # self.channels = channels
        # self.con = con
        super(CNN2, self).__init__()
        ######

        self.conv1 = nn.Conv2d(3, kernels, 3)
        self.conv2 = nn.Conv2d(kernels, kernels, 3)

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(12*12*kernels, neurons)
        self.fc2 = nn.Linear(neurons, 10)
        ######

    def forward(self, features):

        # self.conv = nn.Conv2d(3,kernels,3)
        # self.pool = nn.MaxPool2d(2,2)
        # self.fc1 = nn.Linear(12*12*kernels, neurons)
        # self.fc2 = nn.Linear(neurons, 10)
        x = features
        # for i in range(0, con):
            # print(self.input_size)
            # print(self.channels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

            # channels = kernels
            # input_size = input_size - 3 + 1
            # input_size = round(((input_size-2)/2) +1)

        x = x.view(-1, 12*12*kernels)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x




def plot_accuracy(epochs, train, valid): # Requires epochs as integer, train and valid as lists of accuracy values
    plt.figure()
    lines = plt.plot(epochs, train,'r--', valid,'b')
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(lines[0:2],['Training','Validation'])
    plt.close()
    return lines

def plot_loss(epochs, train, valid): # Requires epochs as integer, train and valid as lists of loss values
    plt.figure()
    lines = plt.plot(epochs, train,'r--', valid,'b')
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(lines[0:2],['Training','Validation'])
    plt.close()
    return lines

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--actf', type=str, choices=['sigmoid', 'relu', 'linear'], default='linear')
    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    eval_every = args.eval_every

def load_basic_model(lr):

    ######
    model = Part3_BasicCNN()
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ######

    return model, loss_fnc, optimizer

# def load_model(lr, kernels, neurons, con, input_size, channels):
def load_model(lr, num):

    ######
    # model = CNN(kernels, neurons, con, input_size, channels)
    if num == 2:
        model = CNN2()
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ######

    return model, loss_fnc, optimizer
def show_image(index, nparray):
    plt.imshow(nparray[index])
    plt.show()

def imshow(img):
    npimg = img.numpy()
   # print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# def evaluate(model, val_loader):
#     total_corr = 0
#     corr = []
#
#     for i, vbatch in enumerate(val_loader):
#         feats, label = vbatch
#         probability = model(feats)
#
#         for j, singleprob in enumerate(probability):
#            # print(singleprob)
#            # print(label[i])
#             labelindex = label[i]
#             predictionindex = torch.argmax(singleprob)
#
#             if labelindex == predictionindex:
#                 corr.append(1.0)
#             else:
#                 corr.append(0.0)
#             total_corr += int(sum(corr))
#         print(total_corr)
#     ######
#
#     return float(total_corr)/len(val_loader.dataset)

def eval(model, loader):
    count = 0
    corr = 0
    for i, batch in enumerate(loader):
        data, label = batch
        label = F.one_hot(label, 10)
        probability = model(data)
        for i,line in enumerate(probability):
            prediction = torch.argmax(line.data)
            l= torch.argmax(label[i])
            #print(prediction)
            corr = corr + (prediction == l).sum().item()
    return corr/len(loader.dataset)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--kernels', type=int, default=10)
parser.add_argument('--neurons', type=int, default=32)
parser.add_argument('--conv', type=int, default=10)
parser.add_argument('--actf', type=str, choices=['sigmoid', 'relu', 'linear'], default='linear')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--input_size', type=int, default=56)

args = parser.parse_args()
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
eval_every = args.eval_every
seed = args.seed
kernels = args.kernels
conv = args.conv
neurons = args.neurons
input_size = args.input_size
channels = args.channels
if __name__ == '__main__':

    # =================================== PART 2 INPUT OWN DATASET AND CHECK 4 IMAGES =========================================== #


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int,default=10)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--eval_every', type=int, default=1)
    # parser.add_argument('--actf', type=str, choices=['sigmoid', 'relu', 'linear'], default='linear')
    # parser.add_argument('--seed', type=int, default=1)
    #
    # args = parser.parse_args()
    #
    # batch_size = args.batch_size
    # lr = args.lr
    # epochs = args.epochs
    # eval_every = args.eval_every
    # seed = args.seed
    # torch.manual_seed(seed)
    #
    # basic_model, loss_fnc, optimizer = load_basic_model(lr)



    ######
    # my_im_folder = "./asl_images_myset"
    #
    # data = torchvision.datasets.ImageFolder(root="./asl_images_myset", transform=transforms.ToTensor())
    # dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, num_workers=2)
    # mean = 0.0
    # std = 0.0
    #
    # for ims, _ in dataloader:
    #     all = ims.size(0)
    #     ims = ims.view(all, ims.size(1), -1)
    #     mean = mean + ims.mean(2).sum(0)
    #     std = std + ims.std(2).sum(0)
    #
    #
    # mean = mean/len(dataloader.dataset)
    # std = std/len(dataloader.dataset)

    # transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean/255, std=std/255)])
    # #onehtransformation = transforms.Compose([transforms.Lambda(lambda x: F.one_hot(x))])
    # #data = torchvision.datasets.ImageFolder(root=my_im_folder, transform = transformation, target_transform = onehtransformation)
    # data = torchvision.datasets.ImageFolder(root=my_im_folder, transform = transformation)
    #
    # trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    #trainloader2 = torch.utils.data.DataLoader(untransformed, batch_size=4, shuffle=True, num_workers=2)

    #test = torchvision.datasets.ImageFolder(root="./asl_images_myset", transform=transformation)
    #testloader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=False, num_workers=2)

    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K')

    # get some random training images
   # dataiter = iter(trainloader2)
    #images, labels = dataiter.next()

    # show images and labels
  #  print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
  #  imshow(torchvision.utils.make_grid(images))
    ######
    # ===================================  =========================================== #

    #
    # basic_model, loss_fnc, optimizer = load_basic_model(lr)
    #
    # t = 0
    #
    # train_accs = []
    # tvals = []
    # valid_accs = []
    # timearray = []
    # loss_array = []
    #
    # st = time.time()
    # print(len(trainloader.dataset))
    # for epoch in range(0, epochs):
    #     accum_loss = 0
    #     for i, batch in enumerate(trainloader):
    #         optimizer.zero_grad()
    #
    #         feats, label = batch
    #         label = F.one_hot(label, 10)
    #
    #         predictions = basic_model(feats)
    #         batch_loss = loss_fnc(input=predictions, target=label.float()).mean()
    #
    #         accum_loss = accum_loss + batch_loss
    #         batch_loss.backward()
    #         optimizer.step()
    #
    #     train_acc = eval(basic_model, trainloader)
    #     end = time.time()
    #
    #     print("Epoch: {}, Step: {} | Loss: {}".format(epoch + 1, t + 1, accum_loss/eval_every))
    #     loss_array = loss_array + [accum_loss]
    #     accum_loss = 0
    #     train_accs = train_accs + [train_acc]
    #     tvals = tvals + [t + 1]
    #     timearray = timearray + [end - st]
    #     t = t + 1
    #
    #     print("Train acc:{}".format(float(train_acc)))
    #     print("Time taken to execute: ", timearray[len(timearray) - 1])
    #
    # plt.plot(tvals, train_accs, 'r')
    # plt.title("Training Accuracy " + " vs." + " Epochs")
    # plt.xlabel("Steps")
    # plt.ylabel("Accuracy")
    # plt.show()
    # plt.close()
    #
    # plt.plot(tvals, loss_array, 'r')
    # plt.title("Loss " + " vs." + " Epochs")
    # plt.xlabel("Steps")
    # plt.ylabel("Loss")
    # plt.show()
    # plt.close()
    #
    # modelinfo = summary(basic_model.cuda(),(3,56,56))


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #




    torch.manual_seed(seed)
    im_folder = "./asl_images"
    mean = 0.0
    std = 0.0
    rawdata = torchvision.datasets.ImageFolder(root=im_folder, transform=transforms.ToTensor())
    rawdata = torch.utils.data.DataLoader(rawdata, shuffle=True)
    for ims, _ in rawdata:
        all = ims.size(0)
        ims = ims.view(all, ims.size(1), -1)
        mean = mean + ims.mean(2).sum(0)
        std = std + ims.std(2).sum(0)

    mean = mean/len(rawdata.dataset)
    std = std/len(rawdata.dataset)
    print("Training mean:", mean)
    print("Training std:", std)


    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    rawdata = torchvision.datasets.ImageFolder(root=im_folder, transform=transformation)

    traindata, validdata = train_test_split(rawdata, test_size=0.2, random_state=seed)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validdata, batch_size=274)

    # mean = 0.0
    # std = 0.0
    #
    #
    # train_mean = 0.0
    # train_std = 0.0
    # valid_mean = 0.0
    # valid_std = 0.0
    #
    # for ims, _ in traindata:
    #     all = ims.size(0)
    #     ims = ims.view(all, ims.size(1), -1)
    #     train_mean = train_mean + ims.mean(2).sum(0)
    #     train_std = train_std + ims.std(2).sum(0)
    #
    #
    # train_mean = train_mean/len(traindata.dataset)
    # train_std = train_std/len(traindata.dataset)
    # print("Training mean:", train_mean)
    # print("Training std:", train_std)
    #
    # for ims, _ in validdata:
    #     all = ims.size(0)
    #     ims = ims.view(all, ims.size(1), -1)
    #     valid_mean = valid_mean + ims.mean(2).sum(0)
    #     valid_std = valid_std + ims.std(2).sum(0)
    #
    #
    # valid_mean = valid_mean/len(validdata.dataset)
    # valid_std = valid_std/len(validdata.dataset)
    #
    # t_transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=train_mean/255, std=train_std/255)])
    # v_transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=valid_mean/255, std=valid_std/255)])
    #
    # trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    # validloader = torch.utils.data.DataLoader(validdata, shuffle=False)

# =================================== Testing Zone =========================================== #




    # model, loss_fnc, optimizer = load_model(lr, kernels, neurons, con, input_size, channels)
    model, loss_fnc, optimizer = load_model(lr, conv)

    t = 0

    train_accs = []
    tvals = []
    valid_accs = []
    timearray = []
    t_loss_array = []
    v_loss_array = []

    st = time.time()
    for epoch in range(0, epochs):
        accum_loss = 0
        v_accum_loss = 0
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()

            feats, label = batch
            label = F.one_hot(label, 10)

            predictions = model(feats)
            batch_loss = loss_fnc(input=predictions, target=label.float()).mean()

            accum_loss = accum_loss + batch_loss
            batch_loss.backward()
            optimizer.step()

        for j, vbatch in enumerate(validloader):

            feats, label = vbatch
            label = F.one_hot(label, 10)

            v_predictions = model(feats)
            v_batch_loss = loss_fnc(input=v_predictions, target=label.float()).mean()

            v_accum_loss = v_accum_loss + v_batch_loss


        train_acc = eval(model, trainloader)
        valid_acc = eval(model, validloader)
        end = time.time()

        print("Epoch: {}, Step: {} | Loss: {} | Valid acc: {} | train acc:{} ".format(epoch + 1, t + 1, batch_loss.item(), valid_acc,train_acc))
        t_loss_array = t_loss_array + [batch_loss.item()]
        v_loss_array = v_loss_array + [v_accum_loss]

        accum_loss = 0
        v_accum_loss = 0
        train_accs = train_accs + [train_acc]
        valid_accs = valid_accs + [valid_acc]

        tvals = tvals + [t + 1]
        timearray = timearray + [end - st]
        t = t + 1

        # print("Time taken to execute: ", timearray[len(timearray) - 1])

    plt.plot(tvals, train_accs, 'r')
    plt.plot(tvals, valid_accs, 'b')
    plt.title("Accuracy " + " vs." + " Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    plt.close()

    plt.plot(tvals, t_loss_array, 'r')
    plt.plot(tvals, v_loss_array, 'b')
    plt.title("Loss " + " vs." + " Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.close()

    modelinfo = summary(model.cuda(),(3,56,56))

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
