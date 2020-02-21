import nltk

nlp_en = spacy.load()
nlp_de = spacy.load()
lang_to_model = {'en': nlp_en, 'de': nlp_de}

# corpus preprocessing
# getting all tags
# getting all named entities

def ner(text):
    pass

def pos(text, lang):
    nlp = lang_to_model[lang]
    tokens = nlp(text)
    features = np.zeros(len(tags))
    tags = [token.pos for token in tokens]
    for i in len(tags):
        features[i] = tags[i]
    return features

def tree():
    pass

def length():
    pass

def fluency():
    pass

def get_features_for_text(text, lang):
    pass

def get_features(segments, lang):
    data = []
    for line in segments:
        f = get_features_for_text(line)
    return features

def main(target_lang, data_dir):
    # Prepare data
    train_data = load_file(data_dir + train_filename, model)
    test_data = load_file(data_dir + test_filename, model)
    train, train_labels = get_features(train_data)
    test, test_labels = get_features(test_data)
    print(
        f'Training shape is {train.shape} and labels is {train_labels.shape}')
    print(f'Testing shape is {test.shape} and labels is {test_labels.shape}')
    # Train classifier
    svc = svm.SVR()
    Cs = [2**k for k in range(-2, 2)]
    params = {'C': Cs}
    clf = GridSearchCV(svc, params)
    model = clf.fit(train, train_labels)
    # Evaluate model.
    test_accuracy = model.score(test, test_labels)
    train_accuracy = model.score(train, train_labels)
    print(f'Parameters used are {model.best_params_}')
    print('Scores:')
    print(f'Accuracy on test set:  {test_accuracy}')
    print(f'Accuracy on train set: {train_accuracy}')


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
