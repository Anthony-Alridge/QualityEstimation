import torch
import transformers as ppb
from torch.utils import data
from torch import nn
from torch import optim
from models.neuralnet import Rishinet3
import pickle

import spacy
import numpy as np
weight = 0

# from nltk import download
# from nltk.corpus import stopwords
#
# nlp_de =spacy.load('de300')
# nlp_en =spacy.load('en300')
#
# #downloading stopwords from the nltk package
# # download('stopwords') #stopwords dictionary, run once
#
# stop_words_en = set(stopwords.words('english'))
# stop_words_de = set(stopwords.words('german'))
#
# def get_sentence_emb(line,nlp,lang):
#   if lang == 'en':
#     text = line.lower()
#     l = [token.lemma_ for token in nlp.tokenizer(text)]
#     l = ' '.join([word for word in l if word not in stop_words_en])
#
#   elif lang == 'de':
#     text = line.lower()
#     l = [token.lemma_ for token in nlp.tokenizer(text)]
#     l= ' '.join([word for word in l if word not in stop_words_de])
#
#   print(l)
#   sen = nlp(l)
#   return sen.vector
#
# def get_embeddings(f,nlp,lang):
#   file = open(f,encoding='utf-8')
#   lines = file.readlines()
#   sentences_vectors =[]
#
#   for l in lines:
#       vec = get_sentence_emb(l,nlp,lang)
#       if vec is not None:
#         # vec = np.mean(vec)
#         sentences_vectors.append(vec)
#       else:
#         print("didn't work :", l)
#         sentences_vectors.append(0)
#
#   return sentences_vectors
#
def pearson(output, target):
    x = torch.flatten(output)
    y = torch.flatten(target)

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
    f = open('../data/en-zh/' + type + '.enzh.src', encoding='utf-8') # Open file on read mode
    lines_en = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    # lines_en = get_embeddings('../data/en-de/' + type + '.ende.src',nlp_en,'en')

    f = open('../data/en-zh/' + type + '.enzh.mt', encoding='utf-8') # Open file on read mode
    lines_de = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    # lines_de = get_embeddings('../data/en-de/' + type + '.ende.mt',nlp_de,'de')
    if type != 'test':
        f = open('../data/en-zh/' + type + '.enzh.scores', encoding='utf-8') # Open file on read mode
        scores = f.read().split("\n") # Create a list containing all lines
        f.close() # Close file

    tokeniser = ppb.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    datas = []
    for i,j in zip(lines_en, lines_de):
      if i is '':break
      # datas.append([i,j])
      datas.append([tokeniser.encode(i,max_length=100,pad_to_max_length=True),tokeniser.encode(j,max_length=100,pad_to_max_length=True)])

    data_tensor = torch.Tensor(datas)

    if type == 'test':
        return data_tensor
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


data_tensor, labels = get_labels_and_data('train')
test_tensor , test_labels = get_labels_and_data('dev')

# f = open("bert_embeddings.pkl",'rb')
# embeddings = np.array(pickle.load(f))
#
# f = open("bert_labels.pkl",'rb')
# labels = np.array(pickle.load(f))
#
# labels = [label[0] for label in labels]
# print(embeddings.shape)



my_dataset = data.TensorDataset(torch.cat([data_tensor,test_tensor],0).cuda(),torch.cat([labels,test_labels]).cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=128,shuffle=True)
print(len(my_dataset))
model = Rishinet3(500000,200)
model.to('cuda')
epochs=20
device=torch.device('cuda')
dtype = torch.long
print_every = 100
optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.1,0.999))

test_dataset = data.TensorDataset(test_tensor.cuda(),test_labels.cuda()) # create your datset#
test_dataloader = data.DataLoader(test_dataset,batch_size=64,shuffle=True)

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
            test(model, test_dataloader)



model.eval()
test_tensor = get_labels_and_data('test')
my_dataset = data.TensorDataset(test_tensor.cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=64)
outs = []
with torch.no_grad():
    for t, (x) in enumerate(my_dataloader):
        x = x[0].to(device=device, dtype=dtype)
        outputs = model(x)
        outs.append(outputs.cpu().detach().numpy())

with open('predictions.txt', 'w') as f:
    for arr in outs:
        for item in arr:
            f.write("%s\n" % item[0])
