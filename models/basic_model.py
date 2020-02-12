import torch
import transformers as ppb
from torch.utils import data
from torch import nn
from torch import optim
from models.neuralnet import Rishinet
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

    datas = []
    for i,j in zip(lines_en, lines_de):
      if i is '':break
      datas.append(tokeniser.encode(i,text_pair=j,truncation_strategy='do_not_truncate',pad_to_max_length=True))
      # datas.append([i,j])
    data_tensor = torch.LongTensor(datas)



    scores_ls = []
    for s in scores:
      if s is None or s is '':break
      scores_ls.append(float(s))

    labels = torch.FloatTensor(scores_ls)
    return data_tensor, labels

data_tensor, labels = get_labels_and_data('train')
# f = open( "data_tensor.pkl", "rb" )
# data_tensor = pickle.load(f)
# pickle.dump(data_tensor,f)
# f = open( "labels.pkl", "rb" )
# labels = pickle.load(f)

# pickle.dump(labels,f)
# print(data_tensor)
# print(data_tensor.shape)
my_dataset = data.TensorDataset(data_tensor,labels) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=2,shuffle=True)

# model = ppb.BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
# model.config.num_labels = 1
model = Rishinet(500000,10,768)
model.to('cpu')

bert = ppb.BertModel.from_pretrained('bert-base-multilingual-cased')
bert.cuda()
epochs=3
device=torch.device('cpu')
dtype = torch.long
print_every = 100
optimizer = optim.Adam(model.parameters(),lr=0.001)
for e in range(epochs):
        for t, (x, y) in enumerate(my_dataloader):
            # print(y)
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float32)
            # print(x.shape)
            # print(x.dtype)
            x = bert(x)[1]
            # print(x)

            outputs = model(x)

            criterion = nn.MSELoss()
            loss = torch.sqrt(criterion(outputs[:2][0], y))
            # print(outputs)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                print()
f = open('model.pkl','wb')
pickle.dump(model,f)
# test_tensor , test_labels = get_labels_and_data('dev')
#
# out = model(test_tensor.cuda())
#
# criterion = nn.MSELoss()
# loss = torch.sqrt(criterion(out[:2][0], test_labels.cuda()))
# print("Acc on test")
# print(loss)