import torch
import transformers as ppb
from torch.utils import data
from torch import nn
from torch import optim
from models.neuralnet import Rishinet3
import pickle

import numpy as np

def loss_func(output,target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return 1 - cost


def get_labels_and_data(type='train'):
    f = open('../data/en-de/' + type + '.ende.src', encoding='utf-8') # Open file on read mode
    lines_en = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    # lines_en = get_embeddings('../data/en-de/' + type + '.ende.src',nlp_en,'en')

    f = open('../data/en-de/' + type + '.ende.mt', encoding='utf-8') # Open file on read mode
    lines_de = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    # lines_de = get_embeddings('../data/en-de/' + type + '.ende.mt',nlp_de,'de')

    f = open('../data/en-de/' + type + '.ende.scores', encoding='utf-8') # Open file on read mode
    scores = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    tokeniser = ppb.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    datas = []
    for i,j in zip(lines_en, lines_de):
      if i is '':break
      datas.append([tokeniser.encode(i,pad_to_max_length=True),tokeniser.encode(j,pad_to_max_length=True)])

    data_tensor = torch.Tensor(datas)

    scores_ls = []
    for s in scores:
      if s is None or s is '':break
      scores_ls.append(float(s))

    labels = torch.FloatTensor(scores_ls)
    return data_tensor, labels

def test(model,dataloader):
    with torch.no_grad():
        model.eval()
        dtype = torch.long
        device = torch.device('cuda')
        running_mse = 0
        running_n = 0
        for t, (x, y) in enumerate(dataloader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float)
            y = y.view(-1, 1)
            outputs = model(x)
            criterion = nn.MSELoss()
            loss = criterion(outputs, y)
            print(loss)
            running_mse += loss.cpu().detach().numpy() * len(x)
            running_n += len(x)

        loss_out = running_mse / running_n
        print("loss is " + str(loss_out))


data_tensor, labels = get_labels_and_data('train')
test_tensor , test_labels = get_labels_and_data('dev')


my_dataset = data.TensorDataset(data_tensor.cuda(),labels.cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=32,shuffle=True)

model = Rishinet3(500000,128)
model.to('cuda')
epochs=20
device=torch.device('cuda')
dtype = torch.long
print_every = 100
optimizer = optim.Adam(model.parameters(),lr=0.00000001)

test_dataset = data.TensorDataset(test_tensor.cuda(),test_labels.cuda()) # create your datset#
test_dataloader = data.DataLoader(test_dataset,batch_size=32,shuffle=True)

for e in range(epochs):
        for t, (x, y) in enumerate(my_dataloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float)
            y = y.view(-1,1)
            outputs = model(x)
            criterion = nn.MSELoss()
            loss = loss_func(outputs, y)

            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                print()

with torch.no_grad():
    model.eval()
    print("validation")
    test(model,test_dataloader)

model.eval()
test_tensor = get_labels_and_data('test')
my_dataset = data.TensorDataset(test_tensor.cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=32)
outs = []
with torch.no_grad():
    for t, (x, y) in enumerate(my_dataloader):
        x = x.to(device=device, dtype=dtype)
        outputs = model(x)
        outs.append(outputs[:2][0].cpu().detach().numpy()[0])

with open('predictions.txt', 'w') as f:
    for item in outs:
        f.write("%s\n" % item)