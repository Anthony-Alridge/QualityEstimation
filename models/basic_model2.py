import torch
import transformers as ppb
from torch.utils import data
from torch import nn
from torch import optim
from models.neuralnet import Rishinet2
import pickle

import spacy
import numpy as np

from nltk import download
from nltk.corpus import stopwords

# nlp_de =spacy.load('de300')
# nlp_en =spacy.load('en300')
#
#
# #downloading stopwords from the nltk package
# download('stopwords') #stopwords dictionary, run once
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
#
#
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

    datas1 = []
    for i,j in zip(lines_en, lines_de):
      if i is '':break
      datas1.append([tokeniser.encode(i,pad_to_max_length=True),tokeniser.encode(j,pad_to_max_length=True)])
      # datas.append([i,j])
    datas = []
    for d in datas1:
        datas.append(d[0] + d[1])
    data_tensor = torch.LongTensor(datas)
    print(data_tensor.shape)



    scores_ls = []
    for s in scores:
      if s is None or s is '':break
      scores_ls.append(float(s))

    labels = torch.FloatTensor(scores_ls)
    return data_tensor, labels

data_tensor, labels = get_labels_and_data('train')

my_dataset = data.TensorDataset(data_tensor.cuda(),labels.cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=16,shuffle=True)

model = Rishinet2(500000,200,2)
model.to('cuda')
epochs=10
device=torch.device('cuda')
dtype = torch.long
print_every = 100
optimizer = optim.Adagrad(model.parameters(),lr=0.0001)
test_tensor , test_labels = get_labels_and_data('dev')

for e in range(epochs):
        for t, (x, y) in enumerate(my_dataloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float)
            y = y.view(-1,1)
            outputs = model(x)
            criterion = nn.MSELoss()
            loss = criterion(outputs, y)
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
            model.to("cuda")
            out = model(test_tensor.cuda())
            criterion = nn.MSELoss()
            lossout = torch.sqrt(criterion(out, test_labels.view(-1,1).cuda()))
            print(lossout)
print("finished")
out = model(test_tensor.cuda())
print(out)
print(out.shape)
criterion = nn.MSELoss()
lossout = torch.sqrt(criterion(torch.flatten(out), test_labels.cuda()))
print("Acc on test")
print(lossout)