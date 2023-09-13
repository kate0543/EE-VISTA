import warnings

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def get_xgboost_classifier(n_classes):
    if n_classes == 2:
        return XGBClassifier(n_estimators=200, verbosity=0, object='binary:logistic', eval_metric=['error'])
    else:
        return XGBClassifier(n_estimators=200, verbosity=0, object='multi:softmax', eval_metric=['merror'])


def hete_list(n_classes):
    classifier_dict = {
        'bernoulli_nb': BernoulliNB(),
        'gaussian_nb': GaussianNB(),
        'xgboost': get_xgboost_classifier(n_classes),
        'random_forest': RandomForestClassifier(n_estimators=200),
        'adaboost': AdaBoostClassifier(n_estimators=200),
        'bagging': BaggingClassifier(n_estimators=200),
        'extra_trees': ExtraTreesClassifier(),

        # 'linear_discriminant_analysis': LinearDiscriminantAnalysis(),
        'random_forest_100': RandomForestClassifier(n_estimators=100),

        'neural_network_mlp': MLPClassifier(),
        'neural_network_bernoulli_rbm': Pipeline(
            steps=[('rbm', BernoulliRBM()), ('logistic', LogisticRegression(solver='newton-cg'))]),

        'k_neighbors_classifier_10': KNeighborsClassifier(n_neighbors=10),
        'k_neighbors_classifier_5': KNeighborsClassifier(n_neighbors=5),
        'k_neighbors_classifier_20': KNeighborsClassifier(n_neighbors=20),

        'logistic_regression': LogisticRegression(solver='newton-cg')
    }

    return classifier_dict


def get_hete_full(n_classes):
    classifier_dict = hete_list(n_classes)
    classifier_list = []
    for classifier_name in classifier_dict:
        classifier_list.append(classifier_dict[classifier_name])
    # Add decision tree classifier to the list
    max_depths = [5, 10, 15]
    min_samples_splits = [0.1, 0.5, 1.0]
    min_samples_leafs = [0.1, 0.25, 0.5]
    for depth in max_depths:
        for split in min_samples_splits:
            for leaf in min_samples_leafs:
                decision_tree_classifier = DecisionTreeClassifier(max_depth=depth, min_samples_split=split,
                                                                  min_samples_leaf=leaf)
                classifier_list.append(decision_tree_classifier)

    layer_sizes = [30, 50, 70]
    for first_layer_size in layer_sizes:
        for second_layer_size in layer_sizes:
            mlp_classifier = MLPClassifier(hidden_layer_sizes=(first_layer_size, second_layer_size))
            classifier_list.append(mlp_classifier)

    return classifier_list


def get_hete_simple(n_classes):
    return [GaussianNB(), KNeighborsClassifier(n_neighbors=5), LogisticRegression(solver='newton-cg'),
            get_xgboost_classifier(n_classes), RandomForestClassifier(n_estimators=200)]
