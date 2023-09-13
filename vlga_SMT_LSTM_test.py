import datetime
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from base_classifier.base_classifiers import get_base_classifiers, EnsembleType
from bin_gcforest import BinaryGcForest, ConcatenateType
from binary_gcforest_problem import Problem
from data_helper import data_folder, file_list
from output_writer import OutputWriter
from surrogate.regressorData import regData
from vlga.vlga_SMT_LSTM import VLGA, SelectionType

"""
    If we pass X_val, y_val to problem_args, fitness will be calculated using validation test,
    otherwise it will be calculated using T-fold CV on training set
"""

NAME = "VLGA_SMT_CHUNK_LSTM"

try:
    # run with file_list[from_id, from_id + 1,..., to_id - 1]
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)

# ------------------------ Parameters ---------------------- #

n_cv_folds = 5
max_n_layers = 100
n_lstm=10
concat_type = ConcatenateType.CONCAT_WITH_ORIGINAL_DATA
n_vlga_generations = 500
n_vlga_population = 100
n_trees = 200
prob_crossover = 0.9
prob_mutation = 0.1
smt_train_iter = 10
smt_estimate_iter = 100

vlga_selection_type = SelectionType.TOURNAMENT
ensemble_type = EnsembleType.HETE
cache_first_layer = False
# ------------------------ Parameters ---------------------- #
def init_classifiers(n_classes):
    return get_base_classifiers(ensemble_type, n_classes, n_trees)


for i_file in range(from_id, to_id):
    file_name = file_list[i_file]
    print(datetime.datetime.now(), ' File {}: '.format(i_file), file_name)
    output_writer = OutputWriter('result/{}/'.format(NAME) + file_name)

    # ---------------------- Prepare Data ---------------------- #
    D_train = np.loadtxt(data_folder + 'train1/' + file_name + '_train1.dat', delimiter=',')
    D_val = np.loadtxt(data_folder + 'val/' + file_name + '_val.dat', delimiter=',')
    D_test = np.loadtxt(data_folder + 'test/' + file_name + '_test.dat', delimiter=',')

    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1].astype(np.int32)
    X_val = D_val[:, :-1]
    Y_val = D_val[:, -1].astype(np.int32)

    X_train_full = np.concatenate((X_train, X_val), axis=0)
    Y_train_full = np.concatenate((Y_train, Y_val))

    X_test = D_test[:, :-1]
    Y_test = D_test[:, -1].astype(np.int32)

    classes = np.unique(np.concatenate((Y_train, Y_val, Y_test)))
    if np.any(classes.astype(np.int32) == 0):
        raise Exception("Labels have to start from 1")

    n_classes = np.size(classes)

    # ------------ Prepare Classifier List ----------------- #
    classifiers, classifiers_str = init_classifiers(n_classes)
    n_classifiers = len(classifiers)

    # ------------ Define Optimization Problem ----------------- #
    common_train_time_start = time.time()

    problem_args = {
        "X": X_train,
        "y": Y_train,
        "X_val": X_val,
        "y_val": Y_val,
        "classifiers": classifiers,
        "n_classes": n_classes,
        "n_cv_folds": n_cv_folds,
        "max_n_layers": max_n_layers,
        "concat_type": concat_type,
        "cache_first_layer": cache_first_layer,
    }

    single_objective_problem = Problem(problem_args)
    regdata = regData(
        chromosome_chunk_size=n_classifiers,
        max_n_chunk=max_n_layers,
        random_state=1
    )
    # ------------------------ VLGA ---------------------------- #
    vlga = VLGA(
        chromosome_chunk_size=n_classifiers,
        max_n_chunk=max_n_layers,
        n_lstm=n_lstm,
        n_generations=n_vlga_generations,
        n_population=n_vlga_population,
        selection_type=vlga_selection_type,
        random_state=1,
        prob_crossover=prob_crossover,
        prob_mutation=prob_mutation,
        smt_train_iter=smt_train_iter,
        smt_estimate_iter=smt_estimate_iter
    )
    '''------------------------ NS-VLGA ----------------------------'''

    best_candidates = vlga.solve(single_objective_problem)

    result_best_candidates = []

    common_train_time_end = time.time()

    map_best_candidate = {}

    for i_bc in range(len(best_candidates)):
        cur_train_time_start = time.time()
    cur_candidate = best_candidates[i_bc].get_config()
    cur_fitness = best_candidates[i_bc].get_fitness()

    cur_config = Problem.candidate_to_config(cur_candidate, n_classifiers, n_classes,
                                             classifiers, n_cv_folds, concat_type)

    cur_result_candidate = {
        "fitness": cur_fitness,
        "n_layers": cur_config["n_layers"],
        "n_classes": n_classes,
        "n_classifiers": n_classifiers,
        "bits": [bits.tolist() for bits in cur_config["bits_list"]],
    }

    result_best_candidates.append(cur_result_candidate)

    '''-------------------------------------------------------------'''

    print('cur_candidate = ', cur_candidate)
    print('cur_fitness = ', cur_fitness)

    # ----------------- Retrain phase -------------------------- #

    # Check if cur_config has already been evaluated
    key = str(cur_candidate)
    if key in map_best_candidate:
        # If cur_config has already been tested, then no need to redo that
        continue
    else:
        wgf = BinaryGcForest(cur_config)

    wgf.fit(X_train_full, Y_train_full)  # retrain model

    cur_train_time_end = time.time()

    # Get more information for paper report -> shouldn't count time of this step
    # Note: recalculated value can be different from
    # the fitness due to the randomness of cv-folds in each layer
    recal_acc = single_objective_problem.fitness(cur_candidate)
    print('recal acc = {}'.format(recal_acc))

    # ------------------ Testing phase ------------------------- #

    test_time_start = time.time()
    prediction = wgf.predict(X_test)

    # --------------------------------------------------------- #
    error = 1 - accuracy_score(Y_test, prediction)
    print('error =', error)
    micro_f1 = f1_score(Y_test - 1, prediction - 1, average='micro')
    print('micro_f1 =', micro_f1)
    macro_f1 = f1_score(Y_test - 1, prediction - 1, average='macro')
    print('macro_f1 =', macro_f1)

    test_time_end = time.time()

    cur_train_time = cur_train_time_end - cur_train_time_start
    test_time = test_time_end - test_time_start

    # Save output to map
    map_best_candidate[key] = [
        cur_train_time,
        test_time,
        prediction,
        error,
        micro_f1,
        macro_f1
    ]

    # ----------------- Writing Output ------------------------- #

    prediction_output = {
        "prediction": prediction.tolist()
    }

    final_output = {
        "error": error,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    time_output = {
        "train_time": cur_train_time + (
                common_train_time_end - common_train_time_start),
        "test_time": test_time
    }

    output_writer.write_output_per_candidate(i_bc, prediction_output, 'predictions', indent=2)
    output_writer.write_output_per_candidate(i_bc, final_output, 'performance', indent=2)
    output_writer.write_output_per_candidate(i_bc, time_output, 'runtime', indent=2)

parameters = {
    "n_cv_folds": n_cv_folds,
    "max_n_layers": max_n_layers,
    "concat_type": concat_type,
    "n_trees": n_trees,
    "classifiers": classifiers_str
}

vlga_output = {
    "generations_average_fitness": vlga.get_result_collector().get_generations_average_fitness(),
    "generations_best_fitness": vlga.get_result_collector().get_generations_best_fitness()
}

output_writer.write_output(result_best_candidates, 'best_candidate', indent=2)
output_writer.write_output(parameters, 'parameters', indent=2)
output_writer.write_output(vlga_output, 'vlga_output', indent=2)
