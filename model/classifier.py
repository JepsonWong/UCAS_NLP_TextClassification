#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "HayesTsai"

from sklearn import svm


class SVMClassifier(object):
    train_ratio = 0.7
    test_ratio = 1 - train_ratio

    def __init__(self, X=None, target=None):
        '''
        :param X: shape([n_samples, n_features])
        :param target: shape([n_samples])
        '''
        self.X = X
        self.target = target
        if self.X is not None and self.target is not None:
            self.train_X = self.X[0:int(len(self.X) * self.train_ratio)]
            self.train_targets = self.target[0:int(len(self.target) * self.train_ratio)]
            self.test_X = self.X[int(len(self.X) * self.train_ratio):]
            self.test_targets = self.target[int(len(self.target) * self.train_ratio):]
        self.model = svm.SVC(C=0.6, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                             probability=False,
                             tol=0.01, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                             decision_function_shape=None,
                             random_state=None)

    def train(self):
        self.model.fit(self.train_X, self.train_targets)

    def test(self):
        predict_targets = self.model.predict(self.test_X)
        from sklearn.metrics import precision_recall_fscore_support as score

        precision, recall, fscore, support = score(self.test_targets, predict_targets)

        import numpy as np
        print('precision: {}'.format(precision))
        print('avg of precision: {}'.format(np.average(precision)))
        print('recall: {}'.format(recall))
        print('avg of recall: {}'.format(np.average(recall)))
        print('fscore: {}'.format(fscore))
        print('avg of fscore: {}'.format(np.average(fscore)))
        print('support: {}'.format(support))
        print('total of support: {}'.format(np.sum(support)))

    def get_model(self):
        return self.model

    def dump(self, X, target, path):
        from sklearn.externals import joblib
        f = open(path, 'w+')
        f.close()
        self.model.fit(X, target)
        joblib.dump(self.model, path)


class GBDTClassifier(object):
    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        gbdt = GradientBoostingClassifier(
            init=None,
            learning_rate=0.1,
            loss='deviance',
            max_depth=3,
            max_features=None,
            max_leaf_nodes=None,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=100,
            random_state=None,
            subsample=1.0,
            verbose=0,
            warm_start=False)
        self.model = gbdt

    def get_model(self):
        return self.model

    def dump(self, X, target, path):
        from sklearn.externals import joblib
        f = open(path, 'w+')
        f.close()
        self.model.fit(X, target)
        joblib.dump(self.model, path)


class Scorer(object):
    def __init__(self, classifier_model, X, targets):
        self.model = classifier_model
        self.X = X
        self.targets = targets

    def show_score(self):
        import sklearn
        if str(sklearn.__version__).startswith('0.18'):
            from sklearn.model_selection import cross_val_score as cvs
        else:
            from sklearn.cross_validation import cross_val_score as cvs

        scores = cvs(self.model, self.X, self.targets, cv=5)
        print("mean of scores is: " + str(scores.mean()))
