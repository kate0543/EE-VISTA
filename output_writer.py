import json
import os

import numpy as np

from binary_gcforest_problem import Problem


class OutputWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def write_output(self, data, file_name, indent=None):
        with open('{}/{}.json'.format(self.output_path, file_name), 'w') as outfile:
            json.dump(data, outfile, indent=indent)

    def write_output_per_candidate(self, i_best, data, file_name, indent=None):
        if not os.path.exists(self.output_path + "/best_{}".format(i_best)):
            os.makedirs(self.output_path + "/best_{}".format(i_best))

        with open('{}/{}.json'.format(self.output_path + "/best_{}".format(i_best), file_name), 'w') as outfile:
            json.dump(data, outfile, indent=indent)

    def write_generation_info(self, gen_P_t, gen_Q_t, gen_R_t, gen_sorted_R_t_indices, n_classifiers):
        if not os.path.exists(self.output_path + "/gen_info"):
            os.makedirs(self.output_path + "/gen_info")

        for i_gen in range(len(gen_P_t)):
            cur_P_t = gen_P_t[i_gen]
            cur_Q_t = gen_Q_t[i_gen]
            cur_R_t = gen_R_t[i_gen]
            cur_sorted_R_t_indices = gen_sorted_R_t_indices[i_gen]

            n_pop = len(cur_P_t)

            # length of map_to_F is len(cur_R_t) = 2*n_pop
            map_to_F = np.zeros(2 * n_pop)
            for i in range(len(cur_sorted_R_t_indices)):
                F = cur_sorted_R_t_indices[i]
                for id in F:
                    map_to_F[id] = i

            cur_pops_P_t = []

            cnt = -1
            for item in cur_P_t:
                cnt += 1
                cur_candidate = item.get_config()
                cur_fitness = item.get_fitness()
                cur_config = Problem.candidate_to_config(cur_candidate, n_classifiers)

                cur_output_candidate = {
                    "fitness": cur_fitness,
                    "bits": [bits.tolist() for bits in cur_config["bits_list"]],
                    "F": map_to_F[cnt]
                }

                cur_pops_P_t.append(cur_output_candidate)

            cur_pops_Q_t = []
            for item in cur_Q_t:
                cnt += 1
                cur_candidate = item.get_config()
                cur_fitness = item.get_fitness()
                cur_config = Problem.candidate_to_config(cur_candidate, n_classifiers)

                cur_output_candidate = {
                    "fitness": cur_fitness,
                    "bits": [bits.tolist() for bits in cur_config["bits_list"]],
                    "F": map_to_F[cnt]
                }

                cur_pops_Q_t.append(cur_output_candidate)

            cur_pops_sorted_R_t = []
            for i in range(len(cur_sorted_R_t_indices)):
                F = cur_sorted_R_t_indices[i]
                for id in F:
                    item = cur_R_t[id]
                    cur_candidate = item.get_config()
                    cur_fitness = item.get_fitness()
                    cur_config = Problem.candidate_to_config(cur_candidate, n_classifiers)

                    cur_output_candidate = {
                        "fitness": cur_fitness,
                        "bits": [bits.tolist() for bits in cur_config["bits_list"]],
                        "F": i,
                        "from": "P_t" if id < n_pop else "Q_t",
                        "id_in_R_t": id
                    }
                    cur_pops_sorted_R_t.append(cur_output_candidate)

            self.write_output(cur_pops_P_t, 'gen_info/gen_{}_P_t'.format(i_gen), indent=2)
            self.write_output(cur_pops_Q_t, 'gen_info/gen_{}_Q_t'.format(i_gen), indent=2)
            self.write_output(cur_pops_sorted_R_t, 'gen_info/gen_{}_sorted_R_t'.format(i_gen), indent=2)
            self.write_output(cur_sorted_R_t_indices, 'gen_info/gen_{}_sorted_R_t_indices'.format(i_gen))
