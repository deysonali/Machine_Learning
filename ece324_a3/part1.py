import torch
import torch.nn as nn

import pandas
import sklearn

import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import dispkernel


# Command Line Arguments

class SNC(nn.Module):

# The constructor builds the pieces of the neural network computation

# we know exactly how many inputsthere are (9), and exactly 1 output
# The nn method Linear, creates a function of the form shown
# in class Z = sum (from i = 0 to 8) of wi *Ii + b
# Calling this constructor allocates those parameters (wi and b) - UNSEEN BY YOU!
# In assignment 2, you created them directly

    def __init__(self):
        super(SNC, self).__init__()
        self.fc1 = nn.Linear(9 , 1)

#  The 'forward' function, is the method that is called when the main class is invoked
#  to compute the classification/prediction.

    def forward(self, I):
        x = self.fc1(I)
        return x

def linear(z): #z is a row of results from the classifier (length = number of input grids)
    y = np.copy(z)
    return y #y is a row with inputs = number of grids

def sigmoid(z): #z is a row of results from the classifier with length = num of input grids
    y = np.copy(z)
    y = 1.0/(1.0 + np.exp(z*(-1)))#should apply the function element wise to every element in z
    return y #row of classifier results with sigmoid function applied.

def relu(z): #z is a row of results from the classifier with length = number of inputs
    y = np.copy(z)
    for i in range(0,np.size(y)):
        if(z[i] > 0.0):
            y[i] = z[i]
        else:
            y[i] = 0.0
    return y #if an element of z is positive, it y remains equal to z at that element. else, it is 0.0.


def calc_z(weights, inp, bias): #for any 1 individual input grid (row array), calculates the z value.
    z = 0
    for i in range(0,np.size(inp)):
        z = z + weights[i]*inp[i]
    z = z + bias
    return z # is a float value corresponding to 1 single input

def loss(z, labels): #takes Z predictions (list of n), and labels (list of n).
    loss = 0.0
    for j in range(0,len(labels)):
        loss = loss + ((z[j]-labels[j])*(z[j]-labels[j]))  #takes the squared error and sums all of the losses
    meanloss = loss/len(labels)
    return meanloss #average value of the loss calculated (float val)


def grad_w(activation, z, labels, data): #z is the full list of predictions, labels are also n long, data is an n (# of ex) by m (# elements in an input) matrix
    grad_w = np.zeros(9) #row of 9 empty 0s
    if activation == "linear":
        for i in range(0, len(labels)): #goes through all n examples
            grad_w = grad_w + 2*(z[i]-labels[i])*data[i] # (element wise) adds the old gradient row to the new gradient row (z-l) is float, el wise x row of input
    elif activation == "sigmoid":
        for i in range(0, len(labels)):
            grad_w = grad_w + 2*(z[i]-labels[i])*(sigmoid(z[i]) - sigmoid(z[i])*sigmoid(z[i]))*data[i]
    elif activation == "relu":
        for i in range(0, len(labels)):
            if(z[i]> 0.0):
                grad_w = grad_w + 2*(z[i]-labels[i])*data[i]
            # else the row stays 0 based on the derivative
    return grad_w/len(labels)

def grad_b(activation, z, labels):
    grad_b = 0.0
    if activation == "linear":
        for i in range(0, len(labels)):
            grad_b = grad_b + 2*(z[i]-labels[i])
    elif activation == "sigmoid":
        for i in range(0, len(labels)):
            grad_b = grad_b + 2*(z[i]-labels[i])*(sigmoid(z[i]) - sigmoid(z[i])*sigmoid(z[i]))
    elif activation == "relu":
        for i in range(0, len(labels)):
            if(z[i] > 0.0):
                grad_b = grad_b + 2*(z[i]-labels[i])*1.0
            #else the row stays 0 based on the derivative
    return grad_b/len(labels)



def update_w(w, grad_w, lrate): #w is an array of size 9, grad_w is a an array of size 9, lrate is a float.
    for i in range(0, len(w)):
        w[i] = w[i] - grad_w[i]*lrate
    return w

def update_b(b, grad_b, lrate): #b, grad_b, lrate are all floats.
    b = b - grad_b*lrate
    return b



def accuracy(predictions,label):

    total_corr = 0

    index = 0
    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1.0
        else:
            r = 0.0
        if (r == label[index].item()):
            total_corr += 1
        index +=1

    return (total_corr/len(label))

def plot_graph(type, epochs, train, valid): #type can be loss or accuracy. Must include both the training and validation data.
    if type == "loss":
        ylab = "Loss"
    elif type == "accuracy":
        ylab = "Accuracy"
    lines = plt.plot(epochs, train,'r--', valid,'b')
    plt.title(ylab+" vs."+" Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(ylab)
    plt.legend(lines[0:2],['Training','Validation'])
    return lines

def gen_colgrids(w):
    im = dispkernel.dispKernel(w,3,3)
    plt.show(im)
    plt.close(im)

def readDataLabels(datafile, labelfile):
  dataset = np.loadtxt(datafile, dtype=np.single, delimiter=',')
  labelset = np.loadtxt(labelfile, dtype=np.single, delimiter=',')

  return dataset, labelset


parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format',default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',default="valid")
parser.add_argument('numtrain', help='number of training samples',type= int,default=200)
parser.add_argument('numvalid', help='number of validation samples',type= int,default=20)
parser.add_argument('-seed', help='random seed', type= int,default=1)
parser.add_argument('-learningrate', help='learning rate', type= float,default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'], default='linear')
parser.add_argument('-numepoch', help='number of epochs', type= int,default=50)

args = parser.parse_args()

trainingfile = args.trainingfile + "data.csv"
traininglabel = args.trainingfile + "label.csv"
numepoch = args.numepoch
print("training data file name: ", trainingfile)
print("training label file name: ", traininglabel)
learningrate = args.learningrate
validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ",args.actfunction)


torch.manual_seed(1)

tdataname = trainingfile
tlabelname = traininglabel

Td, Tl = readDataLabels(tdataname,tlabelname)

numberOfTrainingSamples = len(Td)
print("Number of Training Samples=",numberOfTrainingSamples)
print("Shape of Td=",Td.shape)
print("Shape of Tl=",Tl.shape)

print("Training Data:", Td)
print("Training Labels:", Tl)

# Read Validation Data
vdataname = validdataname
vlabelname = validlabelname

Vd, Vl = readDataLabels(vdataname,vlabelname)

numberOfValidationSamples = len(Vd)
print("Number of Validation Samples=",numberOfValidationSamples)

print("Validation Data:", Vd)
print("Validation Labels:", Vl)

smallNN = SNC()

Tdata = torch.from_numpy(Td)
print("Shape/Size of Tdata = ", Tdata.size())
Tlabel = torch.from_numpy(Tl)
print("Shape/Size of Tlabel = ", Tlabel.size())

Vdata = torch.from_numpy(Vd)
print("Shape/Size of Vdata = ", Vdata.size())
Vlabel = torch.from_numpy(Vl)
print("Shape/Size of Vlabel = ", Vlabel.size())

predict = smallNN(Tdata)

# Let's look at the shape of the output:

print("Shape of Predict", predict.size())

# Call accuracy function to compute how accurate the predictions are
print("Accuracy of Tdata: ", accuracy(predict,Tlabel))

loss_function = torch.nn.MSELoss()   #  mean squared error
optimizer = torch.optim.SGD(smallNN.parameters(),lr=learningrate)

lossRec = []
vlossRec = []
nRec = []
trainAccRec = []
validAccRec = []

for i in range(numepoch):
    # The optimizer, set above, is given the model as input, and so knows where the weights/bias (parameters are)
    # for each epoch, must set the gradients to be zero

    optimizer.zero_grad()

    #  In this epoch, call the model, and have it make a prediction on all of the
    #  instances of the training data
    #  What is happening here?  What is the computation being done on *all* the training data at once?
    #  Recall - each instance of the training data gets multiplied by the weights and bias added
    #  To do it on all 200 instances at once becomes a matrix-vector multiply  (for the weights) and a vector add

    predict = smallNN(Tdata)

    #   From the predictions, and the labels, can compute the loss function
    #   Must 'squeeze' the prediction to eliminate the extra dimension produced
    #   from [200,1] to be [200] so that it aligns with the shape of the labels

    loss = loss_function(input=predict.squeeze(), target=Tlabel.float())

    #   Recall that you computed gradients of the loss with respect to parameters in assignment 2
    #   the following method does that, by tracing backward through all of the computation that
    #   produced the loss value.   PyTorch, when you built your model and then used it to
    #   compute loss, kept track of all of those computations and knows how to compute the gradients
    #   for each (so far, for 'Linear' and 'MSELoss').  We do this with the 'backward' method on
    #   the pyTorch tensor loss

    loss.backward()  # compute the gradients of the weights

    #   Now that the gradients are computed, we ask the optimizer to modify the model parameters (weights and bias)
    #   by taking an optimization step.  (Notice that you never explicitly declared the weights and biases), they
    #   were created when you instantiated smallNN as part of the class NN and the linear function

    optimizer.step()  # this changes the weights and the bias using the learningrate and gradients

    #   compute the accuracy of the model on the training data
    trainAcc = accuracy(predict, Tlabel)

    #   compute the accuracy of the model on the validation data  (don't normally do this every epoch, but is OK here)
    predict = smallNN(Vdata)
    vloss = loss_function(input=predict.squeeze(), target=Vlabel.float())
    validAcc = accuracy(predict, Vlabel)

    print("loss: ", f'{loss:.4f}', " trainAcc: ", f'{trainAcc:.4f}', " validAcc: ", f'{validAcc:.4f}')

    # record data for plotting
    lossRec.append(loss)
    vlossRec.append(vloss)
    nRec.append(i)
    trainAccRec.append(trainAcc)
    validAccRec.append(validAcc)

plt.plot(nRec,lossRec, label='Train')
plt.plot(nRec,vlossRec, label='Validation')
plt.title('Training and Validation Loss vs. epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.close()

plt.plot(nRec,trainAccRec, label='Train')
plt.plot(nRec,validAccRec, label='Validation')
plt.title('Training and Validation Accuracy vs. epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()