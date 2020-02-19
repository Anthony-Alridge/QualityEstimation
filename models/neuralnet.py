import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Rishinet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Rishinet, self).__init__()
        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm1 = nn.LSTM(300, hidden_dim, bidirectional=True)
        self.linear1 = nn.Linear(hidden_dim , 1)

        # self.lstm2 = nn.LSTM(300, hidden_dim, bidirectional=True)
        # self.linear2 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, inputs):
        # print(inputs[:,:,:1])
        # out1, _ = self.lstm1(inputs[:,:1,:])
        # print(inputs.shape)
        out1 = F.tanh(self.linear1(inputs))

        # out2, _ = self.lstm2(inputs[:,1:,:])
        # out2 = self.linear2(out2)
        return out1

class Rishinet2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Rishinet2, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim )
        self.lstm1 = nn.LSTM(embedding_dim , hidden_dim, num_layers=3, bidirectional=False,dropout=0.5,batch_first=True)
        self.linear1 = nn.Linear(1024 * hidden_dim , 50)
        self.linear2 = nn.Linear(50, 1)
        # self.lstm2 = nn.LSTM(300, hidden_dim, bidirectional=True)
        # self.linear2 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, inputs):
        # print(inputs[:,:,:1])
        emb = self.embeddings(inputs)
        out1, _ = self.lstm1(emb)
        out1 = F.tanh(out1.reshape(len(inputs),-1))
        out1 = F.tanh(self.linear1(out1))
        out1 = F.tanh(self.linear2(out1))
        return out1

class Rishinet3(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Rishinet3, self).__init__()
        self.embeddingsE = nn.Embedding(vocab_size, embedding_dim )
        self.convE1 = nn.Conv2d(1,5,5)
        self.maxE1 = nn.MaxPool2d(2)
        self.convE2 = nn.Conv2d(5,7,5)

        self.embeddingsG = nn.Embedding(vocab_size, embedding_dim)
        self.convG1 = nn.Conv2d(1, 5, 5)
        self.maxG1 = nn.MaxPool2d(2)
        self.convG2 = nn.Conv2d(5, 7, 5)

        self.maxPool = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(50750, 300)
        self.linear2 = nn.Linear(300, 1)

    def forward(self, inputs):
        length = len(inputs)
        # print(inputs[:,:,:1])
        embE = self.embeddingsE(inputs[:,:1,:])
        outE = self.convE1(embE)
        outE = self.maxE1(outE)
        outE = self.convE2(outE)

        embG = self.embeddingsG(inputs[:,1:,:])
        outG = self.convG1(embE)
        outG = self.maxG1(outG)
        outG = self.convG2(outG)

        out =  torch.cat([outE,outG],1)

        out = self.maxPool(out)
        out = out.view(length,-1)
        out = F.tanh(self.linear1(out))
        out = F.tanh(self.linear2(out))
        return out
