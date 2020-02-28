# SVM Regression

This model extracts features from the input texts and uses these as input
to a Support Vector Regression algorithm.

Install the following models:
spacy
numpy
sklearn


Before running it will also be necessary to install some spacy dependencies:
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm


## Running the model
Run python main.py --data [path_to_data_directory]


main.py accepts a language argument but the code does not currently support this.
It will default to German and does not support Chinese, so this argument should
be ignored.
