#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np


class FeatureSelection(object):
    def __init__(self, feature_mat, targets):
        """
        :param feature_mat: ndarray
        :param targets: ndarray
        """
        self.feature_mat = feature_mat
        self.targets = targets
        self.feature_filtered = []

    def get_boolean_selection_lst(self):
        return self.feature_filtered


class MISelection(FeatureSelection):
    def __init__(self, feature_mat, targets, mi_threshold=0.06):
        FeatureSelection.__init__(self, feature_mat, targets)
        self.mi_threshold = mi_threshold

    def get_boolean_selection_lst(self):
        """
        Do feature selection
        :return: list of boolean.
        """
        if not self.feature_filtered:
            # Do some filter work here
            from sklearn import metrics as mr
            features = self.feature_mat
            label = self.targets
            x = []
            s = []
            print("MISelection: compute multual begin!")

            for i in range(len(features[0])):
                for j in range(len(features)):
                    x.append(features[j][i])
                s.append(mr.mutual_info_score(label, x))
                x[:] = []
            print("MISelection: compute multual end!")

            for i in range(len(s)):
                if s[i] < self.mi_threshold:
                    self.feature_filtered.append(False)
                else:
                    self.feature_filtered.append(True)
            if not any(self.feature_filtered):
                for i in range(len(self.feature_filtered)):
                    self.feature_filtered[i] = True
        return np.array(self.feature_filtered)


class GBDTSelection(FeatureSelection):
    def get_boolean_selection_lst(self):
        if not self.feature_filtered:
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

            gbdt.fit(self.feature_mat, self.targets)
            self.feature_filtered = [True if item > 0 else False for item in gbdt.feature_importances_]
            if not any(self.feature_filtered):
                for i in range(len(self.feature_filtered)):
                    self.feature_filtered[i] = True
        return np.array(self.feature_filtered)
