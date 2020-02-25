import torch
import transformers as ppb
from torch.utils import data
from torch import nn
from torch import optim
from models.neuralnet import Rishinet3
import pickle
from sklearn import svm, naive_bayes
from sklearn import tree
from sklearn import linear_model

import spacy
import numpy as np
weight = 1


def pearson(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost

def loss_func(output,target):
    x = output
    y = target
    cost = pearson(x,y)
    rmse = torch.sqrt(nn.MSELoss()(x,y))
    return (1 - cost) + weight * rmse


def get_labels_and_data(type='train'):
    f = open('../data/en-de/' + type + '.ende.src', encoding='utf-8') # Open file on read mode
    lines_en = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    # lines_en = get_embeddings('../data/en-de/' + type + '.ende.src',nlp_en,'en')

    f = open('../data/en-de/' + type + '.ende.mt', encoding='utf-8') # Open file on read mode
    lines_de = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    # lines_de = get_embeddings('../data/en-de/' + type + '.ende.mt',nlp_de,'de')

    if type != 'test':
        f = open('../data/en-de/' + type + '.ende.scores', encoding='utf-8') # Open file on read mode
        scores = f.read().split("\n") # Create a list containing all lines
        f.close() # Close file

    tokeniser = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    datas = []
    for i,j in zip(lines_en, lines_de):
      if i is '':break
      datas.append([tokeniser.encode(i,text_pair=j,max_length=100,pad_to_max_length=True)]) #tokeniser.encode(j,max_length=100,pad_to_max_length=True
      # datas.append([i,j])
    data_tensor = torch.Tensor(datas)

    if type == 'test':
        return data_tensor
    scores_ls = []
    for s in scores:
      if s is None or s is '':break
      scores_ls.append(float(s))

    labels = torch.FloatTensor(scores_ls)
    return data_tensor, labels

def validate(model,dataloader):
    with torch.no_grad():
        model.eval()
        dtype = torch.long
        device = torch.device('cuda')
        running_mse = 0
        running_n = 0
        running_pearson = 0
        for t, (x, y) in enumerate(dataloader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float)
            y = y.view(-1, 1)
            outputs = model(x)
            criterion = nn.MSELoss()
            loss = criterion(outputs, y)
            running_mse += loss.cpu().detach().numpy() * len(x)
            running_n += len(x)
            running_pearson += pearson(outputs,y)

        loss_out = running_mse / running_n
        pearson_out = running_pearson / running_n
        print("loss is " + str(loss_out))
        print("pearson is " + str(pearson_out))


# data_tensor, labels = get_labels_and_data('train')
test_tensor , test_labels = get_labels_and_data('dev')


# my_dataset = data.TensorDataset(torch.cat([data_tensor],0).cuda(),torch.cat([labels]).cuda()) # create your datset#
# my_dataloader = data.DataLoader(my_dataset,batch_size=16,shuffle=True)

distilbert = ppb.DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
distilbert.to('cuda')
device=torch.device('cuda')
dtype = torch.long

# embeddings = []
# labels = []
#
# for t, (x, y) in enumerate(my_dataloader):
#     x = x.to(device=device, dtype=dtype).squeeze(1)
#     print(x.shape)
#     outputs = distilbert(x)[0]
#     for output in outputs:
#         embeddings.append(output.cpu().detach().numpy())
#     for l in y:
#         labels.append(l.cpu().detach().numpy())


# f = open("bert_embeddings.pkl",'rb')
# embeddings = np.array(pickle.load(f))
#
# f = open("bert_labels.pkl",'rb')
# labels = np.array(pickle.load(f))
#
# # labels = [label[0] for label in labels]
# print(embeddings.shape)
#
# model = svm.SVR(verbose=True, max_iter=10000,cache_size=100000)
# # model = tree.DecisionTreeRegressor()
# # model = linear_model.BayesianRidge()
# model.fit(embeddings.reshape(7000,100*768),labels)
#
# f = open("bert_model_svr.pkl",'wb')
# pickle.dump(model,f)


f = open("bert_model_svr.pkl",'rb')
model = pickle.load(f)

# test_dataset = data.TensorDataset(test_tensor.cuda(),test_labels.cuda()) # create your datset#
# test_dataloader = data.DataLoader(test_dataset,batch_size=16,shuffle=True)
#
#
# for t, (x,y) in enumerate(test_dataloader):
#     x = x.to(device=device, dtype=dtype).squeeze(1)
#     outputs = distilbert(x)[0]
#     outputs = outputs.cpu().reshape(len(x),100 *768).detach().numpy()
#     y_pred = model.predict(outputs)
#     for (i,j) in zip(y_pred,y):
#         print(str(i) + ", " + str(j))

test_tensor = get_labels_and_data('test')
my_dataset = data.TensorDataset(test_tensor.cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=64)
outs = []
with torch.no_grad():
    for t, (x) in enumerate(my_dataloader):
        x = x[0].to(device=device, dtype=dtype).squeeze(1)
        outputs = distilbert(x)[0]
        outputs = outputs.cpu().reshape(len(x),100 *768).detach().numpy()
        y_pred = model.predict(outputs)
        for i in y_pred:
            outs.append(i)

with open('predictions.txt', 'w') as f:
    for arr in outs:
        f.write("%s\n" % arr)


# print(len(my_dataset))
# model = Rishinet3(500000,200)
# model.to('cuda')
# epochs=20
#
# print_every = 100
# optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.6,0.999))
#

#
# for e in range(epochs):
#         for t, (x, y) in enumerate(my_dataloader):
#             model.train()  # put model to training mode
#             x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
#             y = y.to(device=device, dtype=torch.float)
#             y = y.view(-1,1)
#             outputs = model(x)
#             criterion = nn.MSELoss()
#             loss = loss_func(outputs, y)
#
#             optimizer.zero_grad()
#
#             loss.backward()
#
#             # Update the parameters of the model using the gradients
#             optimizer.step()
#
#             if t % print_every == 0:
#                 print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
#                 #check_accuracy(loader_val, model)
#                 print()
#         with torch.no_grad():
#             model.eval()
#             print("validation")
#             test(model, test_dataloader)
#
#
#
# model.eval()
# test_tensor = get_labels_and_data('test')
# my_dataset = data.TensorDataset(test_tensor.cuda()) # create your datset#
# my_dataloader = data.DataLoader(my_dataset,batch_size=64)
# outs = []
# with torch.no_grad():
#     for t, (x) in enumerate(my_dataloader):
#         x = x[0].to(device=device, dtype=dtype)
#         outputs = model(x)
#         outs.append(outputs.cpu().detach().numpy())
#
# with open('predictions.txt', 'w') as f:
#     for arr in outs:
#         for item in arr:
#             f.write("%s\n" % item[0])
