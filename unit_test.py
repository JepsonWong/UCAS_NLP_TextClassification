#!/usr/bin/env python
# -*- coding: utf-8 -*-


def feature_test():
    pass


def lda_model_test():
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus('./corpus_tiny', encoding='gb18030')
    from feature.ngram_tfidf import NgramTfidf
    unigram_tfidf = NgramTfidf(input_corpus)
    unigram_tfidf.set_stopwords('./resource/stop_words_zh.utf8.txt')
    from model.decomposition import LDADec
    lda = LDADec(unigram_tfidf.get_tf_mat()[0])
    lda.save_doc_topic_mat('./output/unit_test.lda.txt')


def svm_model_test():
    from datasource.input_corpus import InputCorpus
    input_corpus = InputCorpus('./corpus_train', encoding='gb18030')
    from feature.ngram_tfidf import NgramTfidf
    unigram_tfidf = NgramTfidf(input_corpus)
    unigram_tfidf.set_stopwords('./resource/stop_words_zh.utf8.txt')
    from model.decomposition import LDADec
    lda = LDADec(unigram_tfidf.get_tf_mat(top_k=5000)[0], num_topics=500)
    from model.classifier import SVMClassifier
    svm = SVMClassifier(lda.get_doc_topic_mat(), input_corpus.get_filenames_and_targets()[1])
    svm.train()
    svm.test()


def lda_model_local_features_test():
    tf_mat = []
    with open('./output/tf/tf.features.txt') as feature_f:
        feature_f.readline()
        line = feature_f.readline()
        while line:
            line = line.strip()
            tf_mat.append([int(item) for item in line.split(',')])
            line = feature_f.readline()
    from model.decomposition import LDADec
    lda = LDADec(tf_mat)
    lda.save_doc_topic_mat('./test_out_dir/lda/doc_topics_20_from_top_5000_features.txt')


def unit_test():
    feature_test()
    # lda_model_test()
    svm_model_test()
    # lda_model_local_features_test()


if __name__ == '__main__':
    unit_test()
