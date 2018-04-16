import os
import torch
from argparse import ArgumentParser
import torch.nn as nn
from torchtext import data, datasets
import time
from data_utils import makedirs, load_sst, get_args
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

def load_sst_logistic_reg(args):
    # first create two fields for data, text and labels. Check https://github.com/pytorch/text#data for an example
    text = data.Field()
    labels = data.Field(sequential=False, unk_token=None)

    # we first build our dataset with all subtrees to build our vocab
    train_v, valid_v, test_v = datasets.SST.splits(
        text, labels, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')

    text.build_vocab(train_v, valid_v, test_v)
    if args["word_vectors"]:
        if os.path.isfile(args["vector_cache"]):
            text.vocab.vectors = torch.load(args["vector_cache"])
        else:
            text.vocab.load_vectors(args["word_vectors"])
            makedirs(os.path.dirname(args["vector_cache"]))
            torch.save(text.vocab.vectors, args["vector_cache"])
        labels.build_vocab(train_v)

    # Next we build our datasets without all subtrees
    train, valid, test = datasets.SST.splits(text, labels, fine_grained=False, train_subtrees=True,
                                             filter_pred=lambda ex: ex.label != 'neutral')

    return train, valid, test


def logit_clf(inputs, outputs):
    args = get_args()
    args["n_embed"] = len(inputs.vocab)
    args["d_out"] = len(outputs.vocab)

    torch.cuda.set_device(-1)
    train, valid, test = load_sst_logistic_reg(args)

    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

    training_data = [text for text in train.text]
    training_labels = [label for label in train.label]
    validation_data = [text for text in valid.text]
    validation_labels = [label for label in valid.label]
    test_data = [text for text in test.text]
    test_labels = [label for label in test.label]


    bag_of_words = vectorizer.fit_transform(training_data)


    clf = LogisticRegression(dual=True)
    clf.fit(bag_of_words, training_labels)
    predictions = clf.predict(vectorizer.transform(validation_data))

    print(metrics.classification_report(validation_labels,
                                        predictions, target_names=["positive", "negative"]))
    print(metrics.accuracy_score(validation_labels, predictions))


    validation_vectorizer = CountVectorizer(
        tokenizer=lambda doc: doc, lowercase=False)
    validation_vectorizer.fit_transform(validation_data)
    word_coef_lookup = {feature: coef for coef, feature in zip(
        clf.coef_[0], vectorizer.get_feature_names())}
    word_validation_coef_lookup = {
        word: word_coef_lookup[word] for word in validation_vectorizer.vocabulary_ if word in word_coef_lookup}
    return clf, vectorizer


