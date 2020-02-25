import torch
import transformers as ppb
from torch.utils import data
from torch import nn
from torch import optim
from models.neuralnet import Rishinet3

import spacy
from nltk import download
from nltk.corpus import stopwords

nlp_de =spacy.load('de300')
nlp_en =spacy.load('en300')

stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))



tokeniser = ppb.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

l = [token.lemma_ for token in nlp_en.tokenizer("hello")]
l = ' '.join([word for word in l if word not in stop_words_en])
print(nlp_en(l))

l = [token.lemma_ for token in nlp_de.tokenizer("hallo")]
l= ' '.join([word for word in l if word not in stop_words_de])
print(nlp_de(l))