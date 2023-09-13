from enum import Enum

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from gclayer import GcLayer


class ConcatenateType(str, Enum):
    METADATA_ONLY = "metadata_only"
    CONCAT_WITH_ORIGINAL_DATA = "concat_with_original_data"


class BinaryGcForest:

    def __init__(self, config, first_layer=None):
        self.n_layers = config['n_layers']
        self.bits_list = config['bits_list']
        self.classifiers = config['classifiers']
        self.n_cv_folds = config['n_cv_folds']
        self.n_classes = config['n_classes']
        self.n_classifiers = config['n_classifiers']
        self.concat_type = config['concat_type']

        if first_layer is None:
            self.layers = []
        else:
            self.layers = [first_layer]

    def fit(self, X, y):
        XL = np.array(X)
        yL = np.array(y)
        for i_layer in range(self.n_layers):

            # if caching first layer, use it so that
            # classifiers at layer 1 don't need to retrain
            if i_layer == 0 and len(self.layers) > 0:
                layer = self.layers[i_layer]
                layer.set_bits(self.bits_list[i_layer])
                metadata = layer.cross_validation(XL, yL)
            else:
                layer = GcLayer(self.classifiers,
                                self.n_cv_folds,
                                self.n_classes,
                                self.n_classifiers,
                                bits=self.bits_list[i_layer])
                le = LabelEncoder()
                yL = le.fit_transform(yL)
                metadata = layer.cross_validation(XL, yL)
                layer.fit(XL, yL)
                self.layers.append(layer)

            XL = self.get_next_layer_input(metadata, X)

    def predict_proba(self, X):
        XL = np.array(X)
        prob = None
        for i_layer in range(self.n_layers):
            metadata, prob = self.layers[i_layer].predict_proba(XL)
            XL = self.get_next_layer_input(metadata, X)

        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        prediction = np.argmax(prob, axis=1) + 1
        return prediction

    def get_next_layer_input(self, metadata, X):
        """
        """
        input = metadata

        if self.concat_type == ConcatenateType.CONCAT_WITH_ORIGINAL_DATA:
            next_layer_input = np.concatenate((input, X), axis=1)
        elif self.concat_type == ConcatenateType.METADATA_ONLY:
            next_layer_input = np.array(input)
        else:
            raise Exception("Unimplemented exception")

        return next_layer_input

    def test_details(self, X, y):
        """ Use each layer to predict X, return accuracy by comparing to y
        """
        XL = np.array(X)
        layer_errors = []
        layer_prob = None
        for i_layer in range(self.n_layers):
            metadata, layer_prob = self.layers[i_layer].predict_proba(XL)
            prediction = np.argmax(layer_prob, axis=1) + 1
            accuracy = accuracy_score(y, prediction)
            layer_errors.append(1.0 - accuracy)

            XL = self.get_next_layer_input(metadata, X)

        return layer_prob, layer_errors
