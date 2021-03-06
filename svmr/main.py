import spacy
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')
lang_to_model = {'en': nlp_en, 'de': nlp_de}

# Counts the number of named entities in the sentence.
def ner(text):
    nlp = lang_to_model[lang]
    tags_to_idx = corpus_info['tags']
    doc = nlp(text)
    return np.array([len(doc.ents)])

# Returns a 'Bag of Words' style vector where each index corresponds to
# a POS tag and each entry is the frequency of that tag in the given
# sentence.
def pos(text, lang, corpus_info):
    nlp = lang_to_model[lang]
    tags_to_idx = corpus_info['tags']
    tokens = nlp(text)
    features = np.zeros(len(tags_to_idx))
    for token in tokens:
        tag = token.pos_
        if tag in tags_to_idx:
            features[tags_to_idx[tag]] += 1
    return features

# The length of the sentence.
def length(text, lang):
    nlp = lang_to_model[lang]
    tokens = nlp(text)
    return np.array([len(tokens)])

# Apply all features and concatenate arrays together into a single row.
def get_features_for_text(text, lang, corpus_info):
    return np.append(pos(text, lang, corpus_info), length(text, lang))

# Apply features to each segment.
# Each row in return array is features for a single segment
def get_features(src, target, lang, corpus_info):
    data = []
    for line_src, line_target in zip(src, target):
        s = get_features_for_text(line_src, 'en', corpus_info)
        t = get_features_for_text(line_target, lang, corpus_info)
        data.append(np.append(s, t))
    return np.vstack(tuple(data))

# Load the train, dev, or test inputs for a given language
def load_language(language, type = 'train'):
    f = open('../data/en-de/' + type + '.ende.src', encoding='utf-8') # Open file on read mode
    lines_en = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    f = open('../data/en-de/' + type + '.ende.mt', encoding='utf-8') # Open file on read mode
    lines_de = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file
    scores = None
    if type != 'test':
        f = open('../data/en-de/' + type + '.ende.scores', encoding='utf-8') # Open file on read mode
        scores = f.read().split("\n") # Create a list containing all lines
        f.close() # Close file
        scores = [float(score) for score in scores if score != '']
    lines_en = [line for line in lines_en if line != '']
    lines_de = [line for line in lines_de if line != '']
    return lines_en, lines_de, scores

# A pre-processing step to identify all POS tags used in the corpus.
def get_tags(src, target):
    tags = set()
    src_corpus = nlp_en('\n'.join(src))
    target_corpus = nlp_de('\n'.join(target))
    for token in src_corpus:
        tags.add(token.pos_)
    for token in target_corpus:
        tags.add(token.pos_)
    return {tag: i for i, tag in enumerate(tags) }

# Get any global information about the training corpus.
def get_corpus_info(src, target, scores):
    corpus_info = {}
    corpus_info['tags'] = get_tags(src, target)
    return corpus_info

# Return pearsons, mae and rmse stats.
def evaluate(preds, y):
    r = np.corrcoef(preds, y)[0, 1]
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return (r, mae, rmse)

# Apply features to data and train an SVMR model.
# Saves predictions in test set to predictions.txt
def main(target_lang, data_dir):
    # Prepare data
    src, target, scores_ = load_language(target_lang)
    dev, target_dev, scores_dev_ = load_language(target_lang, 'dev')
    test, target_test, _ = load_language(target_lang, 'test')
    scores = np.array(scores_)
    scores_dev = np.array(scores_dev_)
    corpus_info = get_corpus_info(src, target, scores_)
    print('Applying features')
    train = get_features(src, target, target_lang, corpus_info)
    print(f'Training shape is {train.shape} and labels is {scores.shape}')
    dev = get_features(dev, target_dev, target_lang, corpus_info)
    print(f'Validation shape is {dev.shape} and labels is {scores_dev.shape}')
    test = get_features(test, target_test, target_lang, corpus_info)
    print('Test set loaded.')
    # # Train classifier
    svc = SVR()
    Cs = [2**k for k in range(-2, 2)]
    params = {'C': Cs}
    clf = GridSearchCV(svc, params)
    model = clf.fit(train, scores)
    # Evaluate model.
    train_r, train_mse, train_rmse = evaluate(model.predict(train), scores)
    test_r, test_mse, test_rmse = evaluate(model.predict(dev), scores_dev)
    predictions = model.predict(test)
    np.savetxt('predictions.txt', predictions)
    print(f'Parameters used are {model.best_params_}')
    print('Scores:')
    print(f'R on train set: {train_r}, mse: {train_mse}, rmse: {train_rmse}')
    print(f'R on dev set: {test_r}, mse: {test_mse}, rmse: {test_rmse}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SVM model for QE.')
    parser.add_argument(
        '--target',
        default='de',
        help='The shorthand for the target language. Either de (German) or ch (chinese).')
    parser.add_argument(
        '--data_dir',
        default='../data/',
        help='The path to the data directory.')
    args = parser.parse_args()
    main(args.target, args.data_dir)
