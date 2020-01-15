"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""
from models import *
import argparse
import numpy as np
import torch
import torchtext
from torchtext import data
import spacy

torch.manual_seed(3)
np.random.seed(3)

def main(args):
    mods = ['baseline', 'cnn', 'rnn']
    textf = data.Field(sequential=True, tokenize='spacy', include_lengths=True)
    labelf = data.Field(sequential=False, use_vocab=False)
    field = [('TEXT', textf), ('LABEL', labelf)]
    train_ds = data.TabularDataset( path='data/train.tsv', format ='TSV', fields=field)
    __fnc = lambda x: len(x.text)
    textf.build_vocab(train_ds)
    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    textf.vocab.load_vectors(glove)

    baseline = Baseline(args.emb_dim, textf.vocab)
    cnn = CNN(args.emb_dim, textf.vocab, args.num_filt, [2,4])
    rnn = RNN(args.emb_dim, textf.vocab, args.rnn_hidden_dim)

    models = [baseline, cnn, rnn]

    for i, model in enumerate(models):
        models[i] = torch.load('model_{}.pt'.format(mods[i]))
    nlp = spacy.load('en')

    while True:
        print("Enter a sentence")
        x = input()
        tokenized = nlp(x)

        ints = []
        for token in tokenized:
            ints.append(textf.vocab.stoi[str(token)])

        x = torch.LongTensor(ints).unsqueeze(1)
        lengths = torch.IntTensor([len(ints)])
        print("\n")
        i = 0

        for model in models:
            prediction = model(x, lengths)
            # prediction = torch.sigmoid(prediction)
            if prediction >= 0.5:
                type = 'subjective'
            if prediction < 0.5:
                type = 'objective'
            print("Model ", mods[i], ": ", type, " ", "(",float(prediction),")")
            i = i + 1
        print("\n")
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
