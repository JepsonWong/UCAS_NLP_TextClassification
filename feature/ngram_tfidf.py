#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "HayesTsai"

import jieba.posseg as pseg
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Tfidf(object):
    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    def __init__(self, input_corpus):
        self.input_files = input_corpus.get_files()
        self.stop_words = self.STOP_WORDS.copy()
        self.vocab = list()
        self.df = list()
        self.weight = [[]]
        self.tf = [[]]
        self.documents = []

    def get_tfidf_mat(self):
        pass

    def get_tf_mat(self):
        pass

    def __get_docs(self):
        pass

    def set_stopwords(self, stopwords_path):
        from os import path
        from os import getcwd
        _get_abs_path = lambda xpath: path.normpath(path.join(getcwd(), xpath))
        abs_path = _get_abs_path(stopwords_path)
        if not path.isfile(abs_path):
            raise Exception("tfidf: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)


class NgramTfidf(Tfidf):
    def get_tfidf_mat(self, top_k=-1):
        """
        :param top_k: words whose tfidf within top-k will be selected.
        :return: tuple(tfidf_mat, feature_names)
        """
        if not self.documents:
            # 根据 input_corpus 获取文档列表
            self.__get_docs()
        if self.tf == [[]]:
            # 首先获取 term frequency 矩阵, shape:[n_samples, top_k]
            self.get_tf_mat()

        # Transform a count matrix to a normalized tf or tf-idf representation
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(self.tf)

        self.weight = tfidf.toarray()
        return self.__top_k_tfidf(top_k) if top_k > 0 else (self.weight, self.vocab)

    def get_tf_mat(self, top_k=-1):
        if not self.documents:
            self.__get_docs()

        vectorizer = CountVectorizer()
        self.tf = vectorizer.fit_transform(self.documents).toarray()
        self.vocab = vectorizer.get_feature_names()
        # Calc the df begin

        # Calc the df end
        return self.__top_k_tf(top_k) if top_k > 0 else (self.tf, self.vocab)

    def __get_docs(self):
        documents = list()

        for input_file in self.input_files:
            content_text = input_file.get_content()
            words = self.__cut1(content_text)
            documents.append(words)
        self.documents = documents

    def __cut1(self, text):
        words = pseg.cut(text)
        tags = []
        for item in words:
            if item.word in self.stop_words:
                continue
            if item.word.isdigit():
                continue
            tags.append(item.word)
        return " ".join(tags)

    def __cut2(self, text):
        words = pseg.cut(text)
        tags = []
        for item in words:
            if item.word in self.stop_words:
                continue
            if item.word.isdigit():
                continue
            tags.append(item.word)
        bi_grams = []
        for i in range(len(tags) - 1):
            bi_grams.append(tags[i] + tags[i + 1])
        return " ".join(bi_grams)

    def __cut12(self, text):
        return self.__cut1(text) + " " + self.__cut2(text)

    def __top_k_tfidf(self, top_k):
        return self.__top_k_features(self.weight, self.vocab, top_k)

    def __top_k_tf(self, top_k):
        return self.__top_k_features(self.tf, self.vocab, top_k)

    def __top_k_features(self, mat, vec, top_k):
        # select top_k features
        feature_sum_vec = sum(mat)

        sorted_index = list(np.argsort(feature_sum_vec))
        sorted_index.reverse()

        top_k = len(sorted_index) if len(sorted_index) <= top_k else top_k
        new_mat = np.zeros((len(mat), top_k))
        new_vocab = []

        for i in range(top_k):
            new_vocab.append(vec[sorted_index[i]])

        for index_of_doc in range(len(self.tf)):
            for index_of_feature in range(top_k):
                new_mat[index_of_doc][index_of_feature] = mat[index_of_doc][sorted_index[index_of_feature]]
        return new_mat, new_vocab

    def __freq(self, term, document):
        """
        Get freq of term in the given document.
        :param term: str, term.
        :param document: str, split by space.
        :return:
        """
        return document.split().count(term)

    def numDocsContainingFeatures(self, word_list):
        df_vec = [0] * len(word_list)
        for i in range(len(word_list)):
            df_vec[i] = self.numDocsContaining(word_list[i], self.documents)
        return df_vec

    def numDocsContaining(self, word, doclist):
        doccount = 0
        for doc in doclist:
            if self.__freq(word, doc) > 0:
                doccount += 1
        return doccount

    def idf(self, word, doclist):
        n_samples = len(doclist)
        df = self.numDocsContaining(word, doclist)
        return np.log(n_samples / 1 + df)

    def save_tfidf(self, save_to_path, top_k=20):
        if self.weight == [[]]:
            self.get_tfidf_mat()

        new_weight, new_vocab = self.__top_k_tfidf(top_k)

        with open(save_to_path, 'w') as dest_f:
            dest_f.write("file_names,")
            dest_f.write(",".join(new_vocab).encode('utf-8'))
            dest_f.write(",class")
            dest_f.write('\n')
            file_index = 0
            for doc in new_weight:
                dest_f.write(self.input_files[file_index].get_name())
                dest_f.write(',')
                dest_f.write(",".join([str(item) for item in list(doc)]))
                dest_f.write("," + self.input_files[file_index].get_class())
                dest_f.write('\n')
                file_index += 1

    def save_tf(self, save_to_path, top_k=20):
        if self.tf == [[]]:
            self.get_tf_mat()

        new_tf, new_vocab = self.__top_k_tf(top_k)
        with open(save_to_path, 'w') as dest_f:
            dest_f.write("file_names,")
            dest_f.write(",".join(new_vocab).encode('utf-8'))
            dest_f.write(",class")
            dest_f.write('\n')
            file_index = 0
            for doc in new_tf:
                dest_f.write(self.input_files[file_index].get_name())
                dest_f.write(',')
                dest_f.write(",".join([str(int(item)) for item in list(doc)]))
                dest_f.write("," + self.input_files[file_index].get_class())
                dest_f.write('\n')
                file_index += 1
