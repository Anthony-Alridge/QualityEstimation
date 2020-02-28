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
        self.convE1 = nn.Sequential(nn.Conv2d(1,5,5), nn.BatchNorm2d(5), nn.Dropout2d(0.5))
        self.maxE1 = nn.MaxPool2d(2)
        self.convE2 = nn.Sequential(nn.Conv2d(5,10,5), nn.BatchNorm2d(10), nn.Dropout2d(0.5))
        self.maxE2 = nn.MaxPool2d(2)
        self.convE3 = nn.Sequential(nn.Conv2d(10,20,5), nn.BatchNorm2d(20), nn.Dropout2d(0.5))
        self.maxE3 = nn.MaxPool2d(2)
        self.convE4 = nn.Sequential(nn.Conv2d(20, 40, 5), nn.BatchNorm2d(40), nn.Dropout2d(0.5))

        self.embeddingsG = nn.Embedding(vocab_size, embedding_dim)
        self.convG1 = nn.Sequential(nn.Conv2d(1, 5, 5), nn.BatchNorm2d(5), nn.Dropout2d(0.5))
        self.maxG1 = nn.MaxPool2d(2)
        self.convG2 = nn.Sequential(nn.Conv2d(5, 10, 5), nn.BatchNorm2d(10), nn.Dropout2d(0.5))
        self.maxG2 = nn.MaxPool2d(2)
        self.convG3 = nn.Sequential(nn.Conv2d(10,20,5), nn.BatchNorm2d(20), nn.Dropout2d(0.5))
        self.maxG3 = nn.MaxPool2d(2)
        self.convG4 = nn.Sequential(nn.Conv2d(20, 40, 5), nn.BatchNorm2d(40), nn.Dropout2d(0.5))

        self.maxPool = nn.MaxPool2d(2)

        self.linear1 = nn.Sequential(nn.Linear(1280, 300), nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(300, 1), nn.Dropout(0.5))

    def forward(self, inputs):
        length = len(inputs)
        # print(inputs[:,:,:1])
        embE = self.embeddingsE(inputs[:,:1,:])
        outE = F.tanh(self.convE1(embE))
        outE = self.maxE1(outE)
        outE = F.tanh(self.convE2(outE))
        outE = self.maxE2(outE)
        outE = F.tanh(self.convE3(outE))
        outE = self.maxE3(outE)
        outE = F.tanh(self.convE4(outE))

        embG = self.embeddingsG(inputs[:,1:,:])
        outG = F.tanh(self.convG1(embE))
        outG = self.maxG1(outG)
        outG = F.tanh(self.convG2(outG))
        outG = self.maxG2(outG)
        outG = F.tanh(self.convG3(outG))
        outG = self.maxG3(outG)
        outG = F.tanh(self.convG4(outG))

        out =  torch.cat([outE,outG],1)

        out = self.maxPool(out)
        out = out.view(length,-1)
        out = F.tanh(self.linear1(out))
        out = F.tanh(self.linear2(out))
        return out

class Rishinet4(nn.Module):

    def __init__(self, hidden_dim):
        super(Rishinet4, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(hidden_dim , 5000), nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(5000, 2500), nn.Dropout(0.5))
        self.linear3 = nn.Sequential(nn.Linear(2500, 1000), nn.Dropout(0.5))
        self.linear4 = nn.Sequential(nn.Linear(1000, 500), nn.Dropout(0.5))
        self.linear5 = nn.Sequential(nn.Linear(500, 250), nn.Dropout(0.5))
        self.linear6 = nn.Sequential(nn.Linear(250, 100), nn.Dropout(0.5))
        self.linear7 = nn.Linear(100, 1)


    def forward(self, inputs):
        out = F.tanh(self.linear1(inputs))
        out = F.tanh(self.linear2(out))
        out = F.tanh(self.linear3(out))
        out = F.tanh(self.linear4(out))
        out = F.tanh(self.linear5(out))
        out = F.tanh(self.linear6(out))
        out = F.tanh(self.linear7(out))
        return out

class Rishinet5(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Rishinet5, self).__init__()
        self.embeddingsE = nn.Embedding(vocab_size, embedding_dim )
        self.convE1 = nn.Sequential(nn.Conv2d(1,5,5), nn.BatchNorm2d(5), nn.Dropout2d(0.5))
        self.maxE1 = nn.MaxPool2d(2)
        self.convE2 = nn.Sequential(nn.Conv2d(5,10,5), nn.BatchNorm2d(10), nn.Dropout2d(0.5))
        self.maxE2 = nn.MaxPool2d(2)
        self.convE3 = nn.Sequential(nn.Conv2d(10,20,5), nn.BatchNorm2d(20), nn.Dropout2d(0.5))
        self.maxE3 = nn.MaxPool2d(2)
        self.convE4 = nn.Sequential(nn.Conv2d(20, 40, 5), nn.BatchNorm2d(40), nn.Dropout2d(0.5))

        self.embeddingsG = nn.Embedding(vocab_size, embedding_dim)
        self.convG1 = nn.Sequential(nn.Conv2d(1, 5, 5), nn.BatchNorm2d(5), nn.Dropout2d(0.5))
        self.maxG1 = nn.MaxPool2d(2)
        self.convG2 = nn.Sequential(nn.Conv2d(5, 10, 5), nn.BatchNorm2d(10), nn.Dropout2d(0.5))
        self.maxG2 = nn.MaxPool2d(2)
        self.convG3 = nn.Sequential(nn.Conv2d(10,20,5), nn.BatchNorm2d(20), nn.Dropout2d(0.5))
        self.maxG3 = nn.MaxPool2d(2)
        self.convG4 = nn.Sequential(nn.Conv2d(20, 40, 5), nn.BatchNorm2d(40), nn.Dropout2d(0.5))

        self.linear1 = nn.Sequential(nn.Linear(90, 500), nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(500, 250), nn.Dropout(0.5))
        self.linear3 = nn.Sequential(nn.Linear(250, 100), nn.Dropout(0.5))
        self.linear4 = nn.Sequential(nn.Linear(100, 50), nn.Dropout(0.5))


        self.maxPool = nn.MaxPool2d(2)

        self.linear1 = nn.Sequential(nn.Linear(1280, 300), nn.Dropout(0.5))
        self.linear2 = nn.Sequential(nn.Linear(300, 1), nn.Dropout(0.5))

    def forward(self, inputs):
        length = len(inputs)
        print(inputs.shape)
        embE = self.embeddingsE(inputs[:,:1,:])
        outE = F.tanh(self.convE1(embE))
        outE = self.maxE1(outE)
        outE = F.tanh(self.convE2(outE))
        outE = self.maxE2(outE)
        outE = F.tanh(self.convE3(outE))
        outE = self.maxE3(outE)
        outE = F.tanh(self.convE4(outE))

        embG = self.embeddingsG(inputs[:,1:2,:])
        outG = F.tanh(self.convG1(embE))
        outG = self.maxG1(outG)
        outG = F.tanh(self.convG2(outG))
        outG = self.maxG2(outG)
        outG = F.tanh(self.convG3(outG))
        outG = self.maxG3(outG)
        outG = F.tanh(self.convG4(outG))

        outL = F.tanh(self.linear1(inputs[:,2:,:].to(dtype=torch.long)))
        outL = F.tanh(self.linear2(outL))
        outL = F.tanh(self.linear3(outL))
        outL = F.tanh(self.linear4(outL))

        out =  torch.cat([outE,outG, outL],1)

        out = self.maxPool(out)
        out = out.view(length,-1)
        out = F.tanh(self.linear1(out))
        out = F.tanh(self.linear2(out))
        return out