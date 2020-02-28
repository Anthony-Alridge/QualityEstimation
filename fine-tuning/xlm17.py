# -*- coding: utf-8 -*-
"""XLM17.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NxWTg9Fn45brwrMr9OSh2D6i06q5eZB9
"""

from torch.utils import data
import transformers as ppb
import torch
import pandas as pd
import numpy as np
from google.colab import drive
pip install transformers


# Commented out IPython magic to ensure Python compatibility.
drive.mount('/content/gdrive')
!git clone https: // github.com/Anthony-Alridge/QualityEstimation
# %cd 'QualityEstimation'


f = open('data/en-zh/train.enzh.src')  # Open file on read mode
lines_en = f.read().split("\n")  # Create a list containing all lines
f.close()  # Close file

f = open('data/en-zh/train.enzh.mt')  # Open file on read mode
lines_zh = f.read().split("\n")  # Create a list containing all lines
f.close()  # Close file

f = open('data/en-zh/train.enzh.scores')  # Open file on read mode
scores = f.read().split("\n")  # Create a list containing all lines
f.close()  # Close file

tokeniser = ppb.XLMTokenizer.from_pretrained('xlm-mlm-17-1280')

datas = []
for i, j in zip(lines_en, lines_zh):
    if i is '':
        break
    datas.append(tokeniser.encode(i, text_pair=j,
                                  max_length=80, pad_to_max_length=True))
data_tensor = torch.LongTensor(datas)

model = ppb.XLMForSequenceClassification.from_pretrained(
    'xlm-mlm-17-1280', num_labels=1)
model.cuda()

scores_ls = []
for s in scores:
    if s is None or s is '':
        break
    scores_ls.append(float(s))

labels = torch.FloatTensor(scores_ls)


my_dataset = data.TensorDataset(data_tensor, labels)  # create your datset
my_dataloader = data.DataLoader(my_dataset, batch_size=16, shuffle=True)

epochs = 4
device = torch.device('cuda')
dtype = torch.long
print_every = 100
optimizer = torch.optim.Adamax(model.parameters())
for e in range(epochs):
    for t, (x, y) in enumerate(my_dataloader):
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.float)
        outputs = model(x, labels=y)
        loss = outputs[0]
        optimizer.zero_grad()

        loss.backward()

        # Update the parameters of the model using the gradients
        optimizer.step()

        if t % print_every == 0:
            print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
            print()

drive.mount('/content/drive')

path = '/content/drive/My Drive/NLP Chinese Models/xlm_17/'
model.save_pretrained(path)  # save
tokeniser.save_pretrained(path)  # save