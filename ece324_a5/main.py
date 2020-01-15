import torch
import torch.optim as optim
import time

import torchtext
from torchtext import data
import spacy

import argparse
import os

import random

from matplotlib import pyplot as plt

from models import *
from torchsummary import summary


def load_baseline_model(lr, emb_dim, vocab):
    model = Baseline(emb_dim, vocab)
    loss_fnc = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ######

    return model, loss_fnc, optimizer

def load_CNN_model(lr, emb_dim, vocab, num_filt, filter_size):
    model = CNN(emb_dim, vocab, num_filt, filter_size)
    loss_fnc = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ######
    return model, loss_fnc, optimizer

def load_RNN_model(lr, emb_dim, vocab, hidden):
    model = RNN(emb_dim, vocab, hidden)
    loss_fnc = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ######
    return model, loss_fnc, optimizer

def eval(model,loader):
    total = 0
    for i, batch in enumerate(loader):
        batch_input, batch_input_length = batch.text
        predictions = model(batch_input,batch_input_length)
        corr = ((predictions > 0.5).squeeze().long() == batch.label.long())
        total = total + int(corr.sum())
    return float(total)/len(loader.dataset)

def main(args):

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    model = args.model
    emb_dim = args.emb_dim
    rnn_hidden_dim = args. rnn_hidden_dim
    num_filt = args.num_filt

    torch.manual_seed(seed=3)
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    overfit_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='overfit.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
	sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    overfit_iter, val_iter, test_iter = data.BucketIterator.splits(
        (overfit_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    # train_iter, val_iter, test_iter = data.Iterator.splits(
    #     (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
    #     sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    #
    # overfit_iter, val_iter, test_iter = data.Iterator.splits(
    #     (overfit_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
    #     sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data, overfit_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab
    print("Shape of Vocab:", TEXT.vocab.vectors.shape)

    if (model == 'baseline'):
        model, loss_fnc, optimizer = load_baseline_model(lr, emb_dim=emb_dim, vocab=vocab)
    elif (model == 'cnn'):
        model, loss_fnc, optimizer = load_CNN_model(lr, emb_dim, vocab, num_filt, [2,4])
    elif (model == 'rnn'):
        model, loss_fnc, optimizer = load_RNN_model(lr, emb_dim, vocab=vocab, hidden=100)

# OVERFIT
#
#     train_accs = []
#     t_loss_array = []
#     valid_accs = []
#     v_loss_array = []
#     tvals = []
#     timearray = []
#
#     t = 0
#     start = time.time()
#     for epoch in range(0, epochs):
#         accum_loss = 0
#         v_accum_loss = 0
#         for i, batch in enumerate(overfit_iter):
#             optimizer.zero_grad()
#
#             batch_input, batch_input_length = batch.text
#             predictions = model(batch_input, batch_input_length)
#             batch_loss = loss_fnc(input=predictions, target=batch.label.float())
#             accum_loss = accum_loss + batch_loss.item()
#             batch_loss.backward()
#             optimizer.step()
#
#         train_acc = eval(model, overfit_iter)
#         train_accs = train_accs + [train_acc]
#         train_loss = accum_loss/(len(overfit_iter))
#         t_loss_array = t_loss_array + [train_loss]
#         #t_loss_array = t_loss_array + [batch_loss.item()]
#         accum_loss = 0
#
#         for j, v_batch in enumerate(val_iter):
#             v_batch_input, v_batch_input_length = v_batch.text
#             predictions = model(v_batch_input, v_batch_input_length)
#             v_batch_loss = loss_fnc(input=predictions, target=v_batch.label.float())
#             v_accum_loss = v_accum_loss + v_batch_loss.item()
#
#         valid_acc = eval(model, val_iter)
#         valid_accs = valid_accs + [valid_acc]
#         valid_loss = v_accum_loss / (len(val_iter))
#         v_loss_array = v_loss_array + [valid_loss]
#        # v_loss_array = v_loss_array + [v_batch_loss.item()]
#         v_accum_loss = 0
#         end = time.time()
#
#         print(
#             "Epoch: {}, Step: {} | Train Loss: {} | Valid loss: {}| Valid acc: {} | train acc:{} ".format(epoch + 1, t + 1, batch_loss.item(),
#                                                                             v_batch_loss.item(), valid_acc, train_acc))
#         tvals = tvals + [t + 1]
#         timearray = timearray + [end - start]
#         t = t + 1
#
#     print("Time taken to execute: ", timearray[len(timearray) - 1])
#     plt.figure()
#     lines = plt.plot(tvals, train_accs, 'r--', valid_accs,'b')
#     # plt.plot(tvals, train_accs, 'r')
#     # plt.plot(tvals, valid_accs, 'b')
#     plt.title("Accuracy " + " vs." + " Epochs")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend(lines[0:2],['Training','Validation'])
#     plt.show()
#     plt.close()
#
#     plt.figure()
#     lines = plt.plot(tvals, t_loss_array, 'r--', v_loss_array,'b')
#     # plt.plot(tvals, t_loss_array, 'r')
#     # plt.plot(tvals, v_loss_array, 'b')
#     plt.title("Loss " + " vs." + " Epochs")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend(lines[0:2],['Training','Validation'])
#     plt.show()
#     plt.close()

# REGULAR TRAINING
    train_accs = []
    t_loss_array = []
    valid_accs = []
    v_loss_array = []
    tvals = []
    timearray = []


    t = 0
    start = time.time()
    for epoch in range(0, epochs):
        accum_loss = 0
        v_accum_loss = 0
        t_accum_loss = 0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()

            batch_input, batch_input_length = batch.text
            predictions = model(batch_input, batch_input_length)
            batch_loss = loss_fnc(input=predictions, target=batch.label.float())
            accum_loss = accum_loss + batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        train_loss = accum_loss/(len(train_iter))
        train_acc = eval(model, train_iter)
        train_accs = train_accs + [train_acc]
        t_loss_array = t_loss_array + [train_loss]
        # t_loss_array = t_loss_array + [batch_loss.item()]
        accum_loss = 0

        for j, v_batch in enumerate(val_iter):
            v_batch_input, v_batch_input_length = v_batch.text
            predictions = model(v_batch_input, v_batch_input_length)
            v_batch_loss = loss_fnc(input=predictions, target=v_batch.label.float())
            v_accum_loss = v_accum_loss + v_batch_loss.item()

        valid_acc = eval(model, val_iter)
        valid_accs = valid_accs + [valid_acc]
        valid_loss = v_accum_loss/(len(val_iter))
        v_loss_array = v_loss_array + [valid_loss]
        # v_loss_array = v_loss_array + [v_batch_loss.item()]
        v_accum_loss = 0
        end = time.time()

        for k, t_batch in enumerate(test_iter):
            t_batch_input, t_batch_input_length = t_batch.text
            predictions = model(t_batch_input, t_batch_input_length)
            t_batch_loss = loss_fnc(input=predictions, target=t_batch.label.float())
            t_accum_loss = t_accum_loss + t_batch_loss.item()

        test_loss = t_accum_loss / (len(test_iter))
        t_accum_loss = 0

        print(
            "Epoch: {}, Step: {} | Train Loss: {} | Valid loss: {}| test loss: {}| Valid acc: {} | train acc:{} ".format(epoch + 1, t + 1, train_loss,
                                                                            valid_loss, test_loss, valid_acc, train_acc))
        tvals = tvals + [t + 1]
        timearray = timearray + [end - start]
        t = t + 1

    test_accuracy = eval(model, test_iter)
    # torch.save(model, 'model_rnn.pt')
    print("Test accuracy = ", test_accuracy)
    print("Time taken to execute: ", timearray[len(timearray) - 1])
    plt.figure()
    lines = plt.plot(tvals, train_accs, 'r--', valid_accs,'b')
    # plt.plot(tvals, train_accs, 'r')
    # plt.plot(tvals, valid_accs, 'b')
    plt.title("Accuracy " + " vs." + " Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(lines[0:2],['Training','Validation'])
    plt.show()
    plt.close()

    plt.figure()
    lines = plt.plot(tvals, t_loss_array, 'r--', v_loss_array,'b')
    # plt.plot(tvals, t_loss_array, 'r')
    # plt.plot(tvals, v_loss_array, 'b')
    plt.title("Loss " + " vs." + " Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(lines[0:2],['Training','Validation'])
    plt.show()
    plt.close()


    ######

    # 5 Training and Evaluation

    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)



