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

    # f = open('../data/en-de/' + type + '.ende.scores', encoding='utf-8') # Open file on read mode
    # scores = f.read().split("\n") # Create a list containing all lines
    # f.close() # Close file

    tokeniser = ppb.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    datas = []
    for i,j in zip(lines_en, lines_de):
      if i is '':break
      datas.append(tokeniser.encode(i,text_pair=j,truncation_strategy='do_not_truncate',pad_to_max_length=True))
      # datas.append([i,j])
    data_tensor = torch.LongTensor(datas)


    return data_tensor


bert = ppb.BertModel.from_pretrained('bert-base-multilingual-cased')
bert.cuda()

f = open('model.pkl','rb')
model = pickle.load(f)
test_tensor = get_labels_and_data('test')
my_dataset = data.TensorDataset(test_tensor.cuda()) # create your datset#
my_dataloader = data.DataLoader(my_dataset,batch_size=1)
outs = []
for t, x in enumerate(my_dataloader):
    x = bert(x[0])[1]
    out = model(x)
    outs.append(out[:2][0].cpu().detach().numpy()[0])
with open('predictions.txt', 'w') as f:
    for item in outs:
        f.write("%s\n" % item)