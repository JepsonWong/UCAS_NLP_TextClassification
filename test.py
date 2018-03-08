#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from options import options
from options.test_options import TestOptions

LogPrefix = options.TestLogPrefix


def check_args(opt):
    model_path = opt.model_path
    test_dir = opt.test_dir
    suffix_accepted = opt.suffix_accepted
    assert os.path.exists(model_path), model_path + ' does not exist!'
    assert os.path.isdir(test_dir), test_dir + ' does not exist!'
    assert isinstance(suffix_accepted.split(','), list), suffix_accepted + 'should be comma splited!'


def test(opt):
    check_args(opt)
    model_path = opt.model_path
    test_dir = opt.test_dir
    suffix_accepted = opt.suffix_accepted

    # Read documents under test_dir into the list of InputFile
    print(LogPrefix + "Prepare the training corpus begin!")
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus(test_dir, encoding=opt.encoding, suffix_accepted=suffix_accepted)
    print(LogPrefix + "Prepare the training corpus end!")

    # Read the filtered features with pretrained model
    filtered_features_path = model_path + options.FeaturesSaveSuffix
    assert os.path.exists(filtered_features_path), filtered_features_path + 'does not exist! Train model first!'
    features_name = []
    features_df = []

    input_files = input_corpus.get_files()
    n_samples = len(input_files)

    with open(filtered_features_path) as f:
        features_name = f.readline().strip().split(',')
        features_df = [int(item) for item in f.readline().strip().split(',')]
        n_samples += int(f.readline().strip())

    # !!! Generate the feature matrix of InputFile
    from feature.ngram_tfidf import NgramTfidf
    ngram_tfidf = NgramTfidf(input_corpus)
    ngram_tfidf.set_stopwords('./resource/stop_words_zh.utf8.txt')
    import numpy as np

    tf_mat, features_unfiltered = ngram_tfidf.get_tf_mat()
    tf_mat = np.asarray(tf_mat)

    features_mat = np.zeros((len(tf_mat), len(features_df)))

    for i in range(len(tf_mat)):
        # For each doc
        for j in range(len(features_df)):
            # For each feature in current doc
            _tf = tf_mat[i][features_unfiltered.index(features_name[j])] if features_name[
                                                                                j] in features_unfiltered else 0
            _df = features_df[j]
            features_mat[i][j] = _tf * np.log(n_samples / (1 + _df))

    # Load the pretrained model
    from sklearn.externals import joblib
    model = joblib.load(model_path)

    # Do predict
    predicted_Y = model.predict(features_mat)

    # Show the result
    for i in range(len(predicted_Y)):
        print(input_files[i].get_name() + ' is predicted to be class ' + str(predicted_Y[i]) + ' by the model')

    # Calculate the metric
    true_class = [int(input_file_name.get_name().split('-')[0][1:]) for input_file_name in input_files]
    predict_class = [int(predict_c) for predict_c in predicted_Y]
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(np.array(true_class), np.array(predict_class))

    import numpy as np
    print('precision: {}'.format(precision))
    print('avg of precision: {}'.format(np.average(precision)))
    print('recall: {}'.format(recall))
    print('avg of recall: {}'.format(np.average(recall)))
    print('fscore: {}'.format(fscore))
    print('avg of fscore: {}'.format(np.average(fscore)))
    print('support: {}'.format(support))
    print('total of support: {}'.format(np.sum(support)))


if __name__ == '__main__':
    opt = TestOptions().parse_arguments()
    test(opt)
