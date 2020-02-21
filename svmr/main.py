import spacy
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

nlp_en = spacy.load('en_core_web_sm')
nlp_de = spacy.load('de_core_news_sm')
lang_to_model = {'en': nlp_en, 'de': nlp_de}


# Counts the number of named entities in the sentence
def ner(text):
    pass

# Returns array with frequency of each part of speech tag.
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

# The depth and width of the consituency tree.
def tree():
    pass

# The length of the sentence.
def length():
    pass

# The probabality score assigned to the sentece by a language model.
def fluency():
    pass

# Apply all features and concatenate arrays together into a single row.
def get_features_for_text(text, lang, corpus_info):
    return pos(text, lang, corpus_info)

# Apply features to each line.
# Each row in return is feautures for a single line
def get_features(src, target, lang, corpus_info):
    data = []
    for line_src, line_target in zip(src, target):
        s = get_features_for_text(line_src, 'en', corpus_info)
        t = get_features_for_text(line_target, lang, corpus_info)
        data.append(np.append(s, t))
    return np.vstack(tuple(data))

# TODO: make work for chinese
def load_language(language):
    type = 'train'
    f = open('../data/en-de/' + type + '.ende.src', encoding='utf-8') # Open file on read mode
    lines_en = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    f = open('../data/en-de/' + type + '.ende.mt', encoding='utf-8') # Open file on read mode
    lines_de = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file

    f = open('../data/en-de/' + type + '.ende.scores', encoding='utf-8') # Open file on read mode
    scores = f.read().split("\n") # Create a list containing all lines
    f.close() # Close file
    return lines_en, lines_de, scores

def get_corpus_info(src, target):
    tags = set()
    src_corpus = nlp_en('\n'.join(src))
    target_corpus = nlp_de('\n'.join(target))
    for token in src_corpus:
        tags.add(token.pos_)
    for token in target_corpus:
        tags.add(token.pos_)
    corpus_info = {}
    corpus_info['tags'] = {tag: i for i, tag in enumerate(tags) }
    return corpus_info

def evaluate(preds, y):
    sum_x = preds.sum()
    sum_y = y.sum()
    sum_xsq = np.dot(preds, preds)
    sum_ysq = np.dot(y, y)
    sum_xy = np.dot(preds, y)
    error = np.sum(np.abs(preds - y))
    error_sq = np.dot(preds - y, preds - y)
    num_samples += preds.size(0)
    r = num_samples * sum_xy + sum_x * sum_y * np.rsqrt(num_samples * sum_xsq - sum_x ** 2) * np.rsqrt(num_samples * sum_ysq - sum_y ** 2)
    mae = error / num_samples
    rmse = np.sqrt(error_sq / num_samples)
    return (r, mae, rmse)

def main(target_lang, data_dir):
    # Prepare data
    src, target, scores = load_language(target_lang)
    scores = np.array(scores)
    corpus_info = get_corpus_info(src, target)
    # Apply features
    train = get_features(src, target, target_lang, corpus_info)
    # test, test_labels = get_features(test_data)
    print(
        f'Training shape is {train.shape} and labels is {scores.shape}')
    # print(f'Testing shape is {test.shape} and labels is {test_labels.shape}')
    # # Train classifier
    svc = SVR()
    Cs = [2**k for k in range(-2, 2)]
    params = {'C': Cs}
    clf = GridSearchCV(svc, params)
    model = clf.fit(train, scores)
    # # Evaluate model.
    # test_accuracy = model.score(test, test_labels)
    train_r, train_mse, train_rmse = evaluate(model.predict(train), scores)
    print(f'Parameters used are {model.best_params_}')
    print('Scores:')
    #print(f'Accuracy on test set:  {test_accuracy}')
    print(f'R on train set: {train_r}, mse: {train_mse}, rmse: {train_rmse}')


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
