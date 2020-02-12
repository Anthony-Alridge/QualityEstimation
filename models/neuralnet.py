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
        out1 = F.relu(self.linear1(inputs))

        # out2, _ = self.lstm2(inputs[:,1:,:])
        # out2 = self.linear2(out2)
        return out1