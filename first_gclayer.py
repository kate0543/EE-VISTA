import copy

import numpy as np
from sklearn.model_selection import KFold

from gclayer import GcLayer


class FirstGcLayer(GcLayer):
    """
        Way to cache the first layer:
        1. Every classifier in the first layer are trained with (same)original data
        2.
    """

    def __init__(self, classifiers, n_cv_folds, n_classes, n_classifiers, X, y):

        # bits will be set later
        all_bits = np.ones(n_classifiers)
        super().__init__(classifiers, n_cv_folds, n_classes, n_classifiers, all_bits)
        super().fit(X, y)

        self.n_instances = X.shape[0]
        self.cv_prob = [[] for _ in range(n_classifiers)]
        self.is_set_bits = False

    def set_bits(self, bits):
        self.bits = bits
        self.is_set_bits = True
        self.n_active_classifiers = int(np.sum(self.bits))

    def fit(self, X, y):
        print('Already fit when calling constructor')
        pass

    def cross_validation(self, X, y):
        self.check_bits()

        layer_prob = np.zeros((self.n_instances, self.n_classes))
        metadata = np.zeros((self.n_instances, self.n_classes * self.n_active_classifiers))

        # The cross-validation is deterministic
        # because the default value of shuffle=False
        # in KFold() --> [1 2 .. n1][ n1 n1+1 ... n2]...[...n]
        kf = KFold(n_splits=self.n_cv_folds)
        kf_split = kf.split(X)

        cnt_fold = -1
        for train_ids, test_ids in kf_split:
            cnt_fold += 1
            X_train = X[train_ids, :]
            y_train = y[train_ids]

            X_test = X[test_ids, :]

            probs = np.zeros((X_test.shape[0], 0))

            for i_classifier in range(self.n_classifiers):
                if self.bits[i_classifier] == 1:
                    # If this is not the first time -> summon the saved predictions
                    if len(self.cv_prob[i_classifier]) > cnt_fold:
                        prob = self.cv_prob[i_classifier][cnt_fold]
                    else:  # If this is the first time -> fit and predict
                        new_classifier = copy.deepcopy(self.classifiers[i_classifier])
                        new_classifier.fit(X_train, y_train)
                        prob = new_classifier.predict_proba(X_test)
                        self.cv_prob[i_classifier].append(prob)

                    probs = np.concatenate((probs, prob), axis=1)

            metadata[test_ids, :] = probs

        return metadata

    def check_bits(self):
        if self.is_set_bits is False:
            raise Exception("Must set bits for first layer before using")
