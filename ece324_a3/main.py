import argparse
from time import time

import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *
from scipy import signal as sg

from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 20)

""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

# 3.1 YOUR CODE HERE

data = pd.read_csv("data\\adult.csv")
print(data)

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 3.2 YOUR CODE HERE

print("Shape of the data: rows, columns =", data.shape)
print("Column headings are:" , data.columns)
print("First five rows are:")
print(data.head)
print(data["income"].value_counts())

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns

num_rows = data.shape[0]
totalq = 0
for feature in col_names:
    pass
    ######

    # 3.3 YOUR CODE HERE
    totalq = totalq + data[feature].isin(["?"]).sum()
    print("Number of ? in column", feature, " = ", data[feature].isin(["?"]).sum())

print("Total ? found: ", totalq)
    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    # 3.3 YOUR CODE HERE

print("Original Shape:", data.shape)

#deleting rows with ?
for feature in col_names:
    pass
    ######
    data = data[data[feature] !="?"]

print("Shape after deleting rows with missing data:", data.shape)
    ######

# =================================== BALANCE DATASET =========================================== #

    ######

    # 3.4 YOUR CODE HERE

print("Unbalanced counts:", data["income"].value_counts())

#TODO make prettier

# First, based on the value_counts, identify which is underrepresented. All samples used so all of data[data["income"] == ">50K" is used.
# Next, randomly sample n rows with the overrepresented class, with n = number of underrepresented class. that is, n=min(data["income"].value_counts()
# Then mix it up so that there aren't problems with batching

underrepresented = data[data["income"] == ">50K"]
overrepresented = data[data["income"] == "<=50K"]

#keep all of the underrepresented, concatenate with sample of the rest
data = (pd.concat([underrepresented, overrepresented.sample(n=min(data["income"].value_counts()), random_state=0)])).sample(frac=1, random_state=0)
print(data.head)
#Sanity check that the value counts for both classes are now equal
print("Balanced counts:" , data["income"].value_counts())

    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 3.5 YOUR CODE HERE
print(data.describe())
######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    pass
    ######

    print("Value counts for:",feature)
    print(data[feature].value_counts())

    ######

# visualize the first 3 features using pie and bar graphs

######

# 3.5 YOUR CODE HERE
three_features = ['workclass', 'race', 'education']

for feature in three_features:
    labels = data[feature].value_counts().index.tolist()
    values = data[feature].value_counts().values.tolist()
    patches, texts = plt.pie(values, labels=labels, shadow=True, startangle=140)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')

    #plt.show()
    plt.close()

    #binary_bar_chart(data, feature)
    plt.close()
######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
######

def normalize(sample, feature_mean, feature_sdev):
    return (sample-feature_mean)/feature_sdev


# 3.6 YOUR CODE HERE

for feature in data.describe():
   # print(feature)
    mean = data[feature].describe()["mean"]
    sdev = data[feature].describe()["std"]
    #print(mean)
    #print(sdev)
    #print(data[feature])
    data[feature] = normalize(data[feature], mean, sdev)
    #print(data[feature])

continuous = data[['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']].to_numpy()

#print(continuous)
######

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()

income_col = data['income']
income_col = label_encoder.fit_transform(income_col)
######

# 3.6 YOUR CODE HERE

######

oneh_encoder = OneHotEncoder(categories='auto')

######

# 3.6 YOUR CODE HERE
cols = ['workclass', 'race', 'education', 'marital-status', 'occupation', 'relationship', 'gender', 'native-country']
onehot = []

for feature in cols:
    col = label_encoder.fit_transform(data[feature])
    col = col.reshape(-1,1)
    onehot.append((oneh_encoder.fit_transform(col)).toarray())
all_onehot = np.concatenate(onehot,axis=1)

#data = pd.DataFrame(oneh_encoder.fit_transform(data).toarray())
#print(all_onehot)
######


data2 = []
#print(all_onehot.shape[0])
for i in range(all_onehot.shape[0]):
    data2 = data2 + [np.concatenate((continuous[i], all_onehot[i]), axis=None)]

#print(data2)
# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 3.7 YOUR CODE HERE

feat_train, feat_valid, label_train, label_valid = train_test_split(data2, income_col, test_size=0.2, random_state = seed)
######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    ######

    # 4.2 YOUR CODE HERE

    traindata = AdultDataset(feat_train,label_train)
    validdata = AdultDataset(feat_valid,label_valid)
    torch.manual_seed(seed)
    train_loader = DataLoader(traindata,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(validdata,batch_size=batch_size,shuffle=False)

    ######

    return train_loader, val_loader


def load_model(lr):

    ######

    # 4.4 YOUR CODE HERE
    model = MultiLayerPerceptron(103)
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0
    ######

    # 4.6 YOUR CODE HERE

    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch

        prediction = model(feats)
        corr = (prediction > 0.5).squeeze().long() == label.long()
        total_corr += int(corr.sum())
    ######

    return float(total_corr)/len(val_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--actf', type=str, choices=['sigmoid', 'relu', 'linear'], default='linear')
    args = parser.parse_args()

    ######

    # 4.5 YOUR CODE HERE

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    eval_every = args.eval_every


    train_loader, val_loader = load_data(batch_size)
    model, loss_fnc, optimizer = load_model(lr)

    t = 0

    train_accs = []
    tvals = []
    valid_accs = []
    timearray = []
    st = time.time()
    for epoch in range(0,epochs):
        accum_loss = 0
        total_corr = 0
        for i, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()
            predictions = model(feats)
            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
            accum_loss = accum_loss + batch_loss
            batch_loss.backward()
            optimizer.step()
            corr = (predictions>0.5).squeeze().long() == label.long()
            total_corr = total_corr + int(corr.sum())

            end = time.time()
            if (t+1) % (args.eval_every) == 0:
                valid_acc = evaluate(model, val_loader)
                train_acc = evaluate(model, train_loader)

                print("Epoch: {}, Step: {} | Loss: {} | Valid acc: {}".format(epoch+1, t+1, accum_loss/eval_every, valid_acc))
                accum_loss = 0
                valid_accs = valid_accs + [valid_acc]
                train_accs = train_accs + [train_acc]
                tvals = tvals + [t+1]
                timearray = timearray + [end - st]
            t = t + 1

        print("Train acc:{}".format(float(total_corr) / len(train_loader.dataset)))
        print("Time", timearray[len(timearray)-1])
    #lines = plt.plot(tvals, train_accs,'r', valid_accs,'b')
    plt.plot(tvals,sg.savgol_filter(train_accs,9,3,mode="nearest"),'r')
    plt.plot(tvals,sg.savgol_filter(valid_accs,9,3,mode="nearest"), 'b')
    plt.title("Accuracy "+" vs."+" Steps")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend(['Training','Validation'])
    plt.show()
    plt.close()

    plt.plot(timearray,sg.savgol_filter(train_accs,9,3,mode="nearest"),'r')
    plt.plot(timearray,sg.savgol_filter(valid_accs,9,3,mode="nearest"), 'b')
    plt.title("Accuracy "+" vs."+" Time")
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend(['Training','Validation'])
    plt.show()
    plt.close()
    # return lines
    ######


if __name__ == "__main__":
    main()
