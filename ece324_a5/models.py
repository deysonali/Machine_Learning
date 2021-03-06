import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, len=None):
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)
        output = torch.sigmoid(output)
	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return output

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        ######
        pass
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(100,1), nn.Sigmoid())
        ######

    def forward(self, x, len=None):
        ######

        # Section 5.0 YOUR CODE HERE
        pass
        x = self.embed(x)
        x = x.permute(1,0,2)
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1, _ = torch.max(x1, 2)
        x2 = self.conv2(x)
        x2, _ = torch.max(x2, 2)
        x = torch.cat([x1, x2], 1).squeeze()
        x = self.linear(x)
        return x.squeeze()
        ######

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden):
        super(RNN, self).__init__()
        pass
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.rnn = nn.GRU(embedding_dim, hidden)
        self.linear = nn.Sequential(nn.Linear(hidden, 1),nn.Sigmoid())
        ######

    def forward(self, x, len):

        ######

        # Section 6.0 YOUR CODE HERE
        pass
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, len)
        _, h = self.rnn(x)
        x = self.linear(h)
        return x.squeeze()
        ######
