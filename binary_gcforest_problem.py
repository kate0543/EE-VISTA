import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from bin_gcforest import BinaryGcForest
from first_gclayer import FirstGcLayer


class Problem(object):
    """ Problem class defined the targeted problem

    Note: Need to rewrite fitness() function function if change the problem
    """

    def __init__(self, args):

        self.use_cross_validation = True

        if 'X_val' in args:
            self.X_val = args['X_val']
            self.y_val = args['y_val']
            self.use_cross_validation = False

        self.X = args['X']
        self.y = args['y']
        self.classifiers = args['classifiers']
        self.n_classifiers = len(self.classifiers)
        self.n_classes = args['n_classes']
        self.n_cv_folds = args['n_cv_folds']
        self.concat_type = args['concat_type']
        self.cache_first_layer = args['cache_first_layer']
        self.map_fitness = {}

        # dimension: max size of a candidate
        self.dimension = args['max_n_layers'] * self.n_classifiers

        kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=179)
        self.kf_split = list(kf.split(self.X))

        self.__init_first_layer()

    def fitness(self, candidate):
        """ Calculate the fitness score for a given candidate

        Parameters
        ----------
        candidate : determine the structure of a GcForest
            candidate = [c_i] = [0 1 0 0 0 ... 1 1 0]
            candidate.shape = n_layers x n_classifiers

        Returns
        -------
            accuracy : accuracy of GcForest built from the input candidate
        """

        # check if candidate has been evaluated
        key = str(candidate)
        if key in self.map_fitness:
            return self.map_fitness[key]

        config = self.candidate_to_config(candidate, self.n_classifiers,
                                          self.n_classes, self.classifiers,
                                          self.n_cv_folds, self.concat_type)

        if config['n_layers'] == 0 or self.has_zero_layer(config):
            raise Exception("There exists an invalid candidate!")

        if self.use_cross_validation:
            return self.fitness_cross_validation(key, config)
        else:
            return self.fitness_on_val(key, config)

    def fitness_cross_validation(self, key, config):
        y_pred_all = np.zeros(self.X.shape[0])

        cnt = -1
        for train_ids, test_ids in self.kf_split:
            cnt += 1
            X_train_cv = self.X[train_ids, :]
            y_train_cv = self.y[train_ids]

            X_test = self.X[test_ids, :]

            # Note: for each cv-fold, we need a cached first layer
            bin_gc_forest = BinaryGcForest(config, first_layer=self.first_layers[cnt])
            bin_gc_forest.fit(X_train_cv, y_train_cv)

            y_pred = bin_gc_forest.predict(X_test)
            y_pred_all[test_ids] = y_pred

        accuracy = accuracy_score(self.y, y_pred_all)

        self.map_fitness[key] = accuracy
        return accuracy

    def fitness_on_val(self, key, config):

        bin_gc_forest = BinaryGcForest(config, first_layer=self.first_layer)
        bin_gc_forest.fit(self.X, self.y)

        y_pred = bin_gc_forest.predict(self.X_val)

        accuracy = accuracy_score(self.y_val, y_pred)

        self.map_fitness[key] = accuracy
        return accuracy

    def candidate_has_zero_layer(self, candidate):
        config = self.candidate_to_config(candidate, self.n_classifiers)
        if config['n_layers'] == 0:
            return True

        return self.has_zero_layer(config)

    @staticmethod
    def has_zero_layer(config):
        for bits in config['bits_list']:
            if abs(np.sum(bits)) < 0.00001:
                return True

        return False

    @staticmethod
    def candidate_to_config(candidate, n_classifiers, n_classes=None, classifiers=None, n_cv_folds=None,
                            concat_type=None):
        """ Convert a GA candidate to a config determining the structure of GcForest

        Parameters
        ----------
        classifiers : classifier list
        n_classes : number of classes
        n_classifiers : number of classifiers
        n_cv_folds : n_folds used for cross validation
        concat_type : concat with original data or not
        candidate : determine the structure of a GcForest
            candidate = [c_i] = [0 1 0 0 0 ... 1 1 0]
            candidate.shape = n_layers x n_classifiers

        Returns
        -------
        config :
            config = {
                "n_layers" : L,
                "bits_list" : [bits_0, bits_1, ..., bits_L]
                "classifiers" : list of models,
                "n_cv_folds" : n_folds for cross validation,
                "n_classes" : number of classes,
                "n_classifiers" : number of classifiers,
                "concat_type": type of concatenation...
            }
        """
        n_layers = int(candidate.shape[0] * 1. / n_classifiers)
        bits_list = []

        cnt = -1
        for i_layer in range(n_layers):
            bits = np.zeros(n_classifiers)
            for i_classifier in range(n_classifiers):
                cnt += 1
                bits[i_classifier] = candidate[cnt]

            # if np.sum(bits) == 0:
            #     pdb.set_trace()

            bits_list.append(bits)

        config = {
            "n_layers": n_layers,
            "bits_list": bits_list,
            "classifiers": classifiers,
            "n_cv_folds": n_cv_folds,
            "n_classes": n_classes,
            "n_classifiers": n_classifiers,
            "concat_type": concat_type,
        }

        return config

    def __init_first_layer(self):
        cnt = -1

        if self.cache_first_layer:
            if self.use_cross_validation:
                self.first_layers = []
                for train_ids, test_ids in self.kf_split:
                    cnt += 1
                    X_train_cv = self.X[train_ids, :]
                    y_train_cv = self.y[train_ids]

                    self.first_layers.append(
                        FirstGcLayer(self.classifiers,
                                     self.n_cv_folds,
                                     self.n_classes,
                                     self.n_classifiers,
                                     X_train_cv,
                                     y_train_cv)
                    )
            else:
                self.first_layer = FirstGcLayer(self.classifiers,
                                                self.n_cv_folds,
                                                self.n_classes,
                                                self.n_classifiers,
                                                self.X,
                                                self.y)
        else:
            self.first_layers = [None for i in range(self.n_cv_folds)]
            self.first_layer = None
