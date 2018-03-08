#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from options import options
from options.train_options import TrainOptions


def check_args(opt):
    corpus_root = opt.corpus_root
    which_filter = opt.which_filter.lower()
    which_classifier = opt.which_classifier.lower()
    assert which_filter in ['mi', 'gbdt'], 'Only mi or gbdt allowed!'
    assert which_classifier in ['svm', 'gbdt'], 'Only svm or gbdt allowed!'
    assert os.path.isdir(corpus_root), corpus_root + ' does not exist!'
    assert 1 > opt.mi_threshold > 0, 'mi_threshold belongs to (0,1)!'


def save_features_df(df_vec, vocab_vec, n_samples, path_to_save):
    with open(path_to_save, 'a') as f:
        f.write(",".join(vocab_vec).encode('utf-8'))
        f.write('\n')
        f.write(','.join([str(item) for item in df_vec]))
        f.write('\n')
        f.write(str(n_samples))
        f.write('\n')


def train(opt):
    # Prepare the training corpus
    print(options.TrainLogPrefix + "Prepare the training corpus begin!")
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus(opt.corpus_root, encoding=opt.encoding)
    print(options.TrainLogPrefix + "Prepare the training corpus end!")

    # Get the basic tfidf features
    print(options.TrainLogPrefix + "Get the basic tfidf features begin!")
    from feature.ngram_tfidf import NgramTfidf
    ngram_tfidf = NgramTfidf(input_corpus)
    ngram_tfidf.set_stopwords('./resource/stop_words_zh.utf8.txt')
    import numpy as np
    tfidf_mat, features = ngram_tfidf.get_tfidf_mat(top_k=opt.tfidf_top_k)
    tfidf_mat = np.asarray(tfidf_mat)
    features = np.asarray(features)
    targets = np.asarray(input_corpus.get_filenames_and_targets()[1])
    print(options.TrainLogPrefix + "Get the basic tfidf features end!")

    # Do feature selection
    print(options.TrainLogPrefix + "Do feature selection begin!")
    if opt.which_filter == 'mi':
        from feature.feature_selection import MISelection as FeatureSelection
        feature_selector = FeatureSelection(tfidf_mat, targets, mi_threshold=opt.mi_threshold)
    else:
        from feature.feature_selection import GBDTSelection as FeatureSelection
        feature_selector = FeatureSelection(tfidf_mat, targets)
    boolean_selection_index = feature_selector.get_boolean_selection_lst()
    filtered_tfidf_mat = tfidf_mat[:, boolean_selection_index]
    filtered_features = features[boolean_selection_index]
    print(options.TrainLogPrefix + "Do feature selection end!")

    # Training model
    print(options.TrainLogPrefix + "Training model begin!")
    if opt.which_classifier == 'svm':
        from model.classifier import SVMClassifier as Classifier
    else:
        from model.classifier import GBDTClassifier as Classifier
    classifier_model = Classifier()
    from model.classifier import Scorer
    scorer = Scorer(classifier_model.get_model(), filtered_tfidf_mat, targets)
    print(options.TrainLogPrefix + "Training model end!")
    scorer.show_score()

    # Save the model
    model_save_path = opt.path_to_save_model
    from utils import util
    util.mkdirs('/'.join(model_save_path.split('/')[:-1]))
    classifier_model.dump(filtered_tfidf_mat, targets, model_save_path)
    print(options.TrainLogPrefix + 'model save to ' + model_save_path)

    # Save the filtered features
    filtered_features_save_path = opt.path_to_save_model + options.FeaturesSaveSuffix
    df_vec = ngram_tfidf.numDocsContainingFeatures(filtered_features)
    save_features_df(df_vec, filtered_features, len(tfidf_mat), filtered_features_save_path)


if __name__ == '__main__':
    # Parse arguments
    opt = TrainOptions().parse_arguments()
    train(opt)
