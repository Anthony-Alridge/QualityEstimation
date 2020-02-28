# -*- coding: utf-8 -*-
"""Model Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LCsSGpZ8Efrb8QxlmjeFvIDUXsPytGHq
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


f = open('data/en-zh/dev.enzh.src')  # Open file on read mode
lines_en = f.read().split("\n")  # Create a list containing all lines
f.close()  # Close file

f = open('data/en-zh/dev.enzh.mt')  # Open file on read mode
lines_zh = f.read().split("\n")  # Create a list containing all lines
f.close()  # Close file

f = open('data/en-zh/dev.enzh.scores')  # Open file on read mode
scores = f.read().split("\n")  # Create a list containing all lines
f.close()  # Close file

path = '/content/gdrive/My Drive/NLP Chinese Models/distilbert/'
model = ppb.DistilBertForSequenceClassification.from_pretrained(
    path)  # re-load
tokenizer = ppb.DistilBertTokenizer.from_pretrained(path)  # re-load
model.cuda()

scores_ls = []
for s in scores:
    if s is None or s is '':
        break
    scores_ls.append(float(s))

labels = torch.FloatTensor(scores_ls)

datas = []
for i, j in zip(lines_en, lines_zh):
    if i is '':
        break
    datas.append(tokenizer.encode(i, text_pair=j,
                                  max_length=80, pad_to_max_length=True))
data_tensor = torch.LongTensor(datas)

my_dataset = data.TensorDataset(data_tensor, labels)  # create your datset
my_dataloader = data.DataLoader(my_dataset, batch_size=32, shuffle=False)

model.train()
device = torch.device('cuda')
dtype = torch.long
outs = []
for t, (x, y) in enumerate(my_dataloader):
    with torch.no_grad():
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.float)
        outputs = model(x)
        outputs = torch.flatten(outputs[0]).cpu()
        outputs = outputs.numpy()
        outs.append(outputs)

with open('/content/gdrive/My Drive/NLP Chinese Models/distil_predictions.txt', 'a') as f:
    for a in outs:
        for x in a:
            print(x, file=f)