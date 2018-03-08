#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "HayesTsai"

from sklearn.decomposition import LatentDirichletAllocation

class LDADec(object):
    def __init__(self, tf_mat, num_topics=20, max_iter=5):
        self.tf = tf_mat
        self.num_topics = num_topics
        self.max_iter = max_iter
        self.doc_topic_mat = [[]]

    def get_doc_topic_mat(self):
        if self.doc_topic_mat == [[]]:
            lda = LatentDirichletAllocation(
                n_topics=self.num_topics,
                max_iter=self.max_iter,
                learning_method='online',
                learning_offset=50.,
                random_state=0
            )
            lda.fit(self.tf)
            # doc_topic_distr : shape=(n_samples, n_topics),Document topic distribution for tf.
            doc_topic_distr = lda.transform(self.tf)
            self.doc_topic_mat = doc_topic_distr
        return self.doc_topic_mat

    def save_doc_topic_mat(self, path_to_save):
        if self.doc_topic_mat == [[]]:
            self.get_doc_topic_mat()
        with open(path_to_save, 'a') as dest_f:
            for doc in self.doc_topic_mat:
                dest_f.write(",".join(['%f' % item for item in doc]))
                dest_f.write('\n')
