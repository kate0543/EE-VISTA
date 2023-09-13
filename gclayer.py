import copy

import numpy as np
from sklearn.model_selection import KFold


class GcLayer:
    def __init__(self, classifiers, n_cv_folds, n_classes, n_classifiers, bits=None):
        """
        Parameters
        ----------
        bits: [0 1 0 1 1 ... 1], shape = 1 x n_classifiers
                bits[i] = 1 if classifiers[i] is active
        """

        self.bits = bits
        self.classifiers = classifiers
        self.trained_models = []
        self.n_cv_folds = n_cv_folds
        self.n_classes = n_classes
        self.n_classifiers = n_classifiers

        self.n_active_classifiers = int(np.sum(self.bits))

    def fit(self, X, y):
        """ train all models i that bits[i] = 1
        """

        for i_classifier in range(self.n_classifiers):
            if self.bits[i_classifier] == 1:
                new_classifier = copy.deepcopy(self.classifiers[i_classifier])
                new_classifier.fit(X, y)
                self.trained_models.append(new_classifier)
            else:
                # force the indices of trained_models and classifiers to be the same
                self.trained_models.append(None)

    def cross_validation(self, X, y):
        """ Apply K-folds cross validation on (X, y)
            to obtain the metadata (use active classifiers only)

        Parameters
        ----------
        X : samples
        y : labels

        Returns
        -------
        metadata : n_instances x (n_classes * n_active_classifiers)
        """

        n_instances = X.shape[0]
        metadata = np.zeros((n_instances, self.n_classes * self.n_active_classifiers))

        kf = KFold(n_splits=self.n_cv_folds)
        kf_split = kf.split(X)

        for train_ids, test_ids in kf_split:
            X_train = X[train_ids, :]
            y_train = y[train_ids]

            X_test = X[test_ids, :]

            probs = np.zeros((X_test.shape[0], 0))

            for i_classifier in range(self.n_classifiers):
                if self.bits[i_classifier] == 1:
                    new_classifier = copy.deepcopy(self.classifiers[i_classifier])
                    new_classifier.fit(X_train, y_train)
                    prob = new_classifier.predict_proba(X_test)

                    probs = np.concatenate((probs, prob), axis=1)

            metadata[test_ids, :] = probs

        return metadata

    def predict(self, X):
        """ use trained_models to predict for X

        Parameters
        ---------
        X :

        Returns
        -------
        """
        raise Exception("Unimplemented exception!")

    def predict_proba(self, X):
        """ use trained_models to predict probabilities for X

        Parameters
        ----------
        X : unlabeled data (n_instances x n_features)

        Returns
        -------
        metadata : n_instances x (n_classes * n_active_classifiers)
        layer_prob : n_instances x n_classes
        """
        n_instances = X.shape[0]
        layer_prob = np.zeros((n_instances, self.n_classes))
        metadata = np.zeros((n_instances, 0))

        for i_classifier in range(self.n_classifiers):
            if self.bits[i_classifier] == 1:
                # Note: indices of trained_models and classifiers are the same
                prob = self.trained_models[i_classifier].predict_proba(X)

                layer_prob = layer_prob + prob
                metadata = np.concatenate((metadata, prob), axis=1)

        layer_prob = layer_prob / len(self.trained_models)
        return metadata, layer_prob
