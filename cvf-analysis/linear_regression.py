import os
import copy
import json

import numpy as np
import pandas as pd

from functools import lru_cache
from cvf_analysis import CVFAnalysis, logger


class LinearRegressionFullAnalysis(CVFAnalysis):
    results_prefix = "linear_regression"
    results_dir = os.path.join("results", results_prefix)

    def __init__(self, graph_name, graph) -> None:
        super().__init__(graph_name, graph)
        self.learning_rate = 0.001
        self.slope_step_decimals = 1
        self.slope_step = 1 / (10**self.slope_step_decimals)
        self.min_slope = 0
        # self.max_slope = 4
        # self.actual_m = 3.0834764453827943
        # self.actual_b = -82.57574306316957
        # self.no_of_nodes = 4
        # self.df = pd.read_csv(
        #     "/home/agaru/research/cvf-python/linear_regression/SOCR-HeightWeight.csv"
        # )
        # self.df.rename(
        #     columns={"Height(Inches)": "X", "Weight(Pounds)": "y"}, inplace=True
        # )
        # self.doubly_stochastic_matrix_config = [
        #     [1 / 3, 1 / 6, 1 / 6, 1 / 3],
        #     [1 / 6, 1 / 6, 1 / 3, 1 / 3],
        #     [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        #     [1 / 3, 1 / 3, 1 / 6, 1 / 6],
        # ]
        self.max_slope = 1.0
        self.actual_m = 0.9
        self.actual_b = -0.11847322643445737
        self.no_of_nodes = 3
        self.df = pd.read_csv(
            "/home/agaru/research/cvf-python/linear_regression/random-data.csv"
        )
        self.doubly_stochastic_matrix_config = [
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
            [1 / 6, 2 / 3, 1 / 6],
        ]
        self.node_data_partitions = np.array_split(self.df, self.no_of_nodes)
        for i, node_data in enumerate(self.node_data_partitions):
            self.df.loc[node_data.index, "node"] = i

        self.df["partition"] = -1
        for i in range(self.no_of_nodes):
            node_filter = self.df["node"] == i
            node_df = self.df[node_filter]
            partitions = self.__gen_test_data_partition_frm_df(
                self.no_of_nodes, node_df
            )
            for i, p in enumerate(partitions):
                self.df.loc[self.df.index.isin(p.index.values), "partition"] = i

    def __gen_test_data_partition_frm_df(self, partitions, df):
        shuffled = df.sample(frac=1)
        result = np.array_split(shuffled, partitions)
        return result

    def _start(self):
        self._gen_configurations()
        self._find_invariants()
        self._init_pts_rank()
        self._find_program_transitions_n_cvfs()
        # self.__save_pts_to_file()
        # self._rank_all_states()
        # self._gen_save_rank_count()
        # self._calculate_pts_rank_effect()
        # self._calculate_cvfs_rank_effect()
        # self._gen_save_rank_effect_count()
        # self._gen_save_rank_effect_by_node_count()

    def _gen_configurations(self):
        self.configurations = {tuple([self.min_slope for _ in range(len(self.nodes))])}
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for node_pos in self.nodes:
            config_copy = copy.deepcopy(self.configurations)
            for i in np.round(
                np.arange(
                    self.min_slope + self.slope_step,
                    self.max_slope + self.slope_step,
                    self.slope_step,
                ),
                2,
            ):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def __get_adjusted_value(self, value):
        if value > self.max_slope:
            return self.max_slope

        if value < self.min_slope:
            return self.min_slope

        result = value

        if result / self.slope_step != 0:
            result = (result // self.slope_step) * self.slope_step

        if (value - result) > self.slope_step / 2:
            result = result + self.slope_step

        return result

    def _find_invariants(self):
        min_loss_sum = 1000000
        min_loss_sum_state = None
        for state in self.configurations:
            temp = 0
            for node, m in enumerate(state):
                node_df = self.__get_node_data_df(node)
                X_node = node_df["X"].array
                y_node = node_df["y"].array
                params = {"m": m, "c": 0}
                y_node_pred = self.__forward(X_node, params)
                loss = self.__loss_fn(y_node, y_node_pred)
                temp += loss

            if abs(temp) < min_loss_sum:
                min_loss_sum = abs(temp)
                min_loss_sum_state = state

            # for m in state:
            #     if not (
            #         self.actual_m - self.slope_step / 2
            #         < m
            #         <= self.actual_m + self.slope_step / 2
            #     ):
            #         break
            # else:
            #     self.invariants.add(state)

        self.invariants.add(min_loss_sum_state)
        print("Invariants", self.invariants)
        logger.info("No. of Invariants: %s", len(self.invariants))

    def __forward(self, X, params):
        return params["m"] * X + params["c"]

    def __loss_fn(self, y, y_pred):
        N = len(y)
        return (1 / N) * sum((y[i] - y_pred[i]) ** 2 for i in range(N))

    def __gradient_m(self, X, y, y_pred):
        N = len(y)
        return (-2 / N) * np.sum(X * (y - y_pred))

    def __get_node_data_df(self, node_id):
        return self.df[self.df["node"] == node_id]

    def _is_program_transition(
        self, perturb_pos, start_state, dest_state, grad_m
    ) -> bool:
        perturbed_m = dest_state[perturb_pos]
        doubly_st_mt = self.doubly_stochastic_matrix_config[perturb_pos]
        new_m = (
            sum(frac * start_state[i] for i, frac in enumerate(doubly_st_mt))
            - self.learning_rate * grad_m
        )
        delta = new_m - start_state[perturb_pos]
        ad_new_m = self.__get_adjusted_value(new_m)
        ad_new_m = np.round(ad_new_m, self.slope_step_decimals)
        return delta, ad_new_m == perturbed_m

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        all_slope_values = set(
            np.round(
                np.arange(
                    self.min_slope, self.max_slope + self.slope_step, self.slope_step
                ),
                2,
            )
        )

        deltas = {}
        for position, val in enumerate(start_state):
            node_df = self.__get_node_data_df(position)
            X_node = node_df["X"].array
            y_node = node_df["y"].array
            params = {"m": val, "c": 0}
            y_node_pred = self.__forward(X_node, params)
            grad_m = self.__gradient_m(X_node, y_node, y_node_pred)
            print(position, start_state)
            deltas[position] = {}

            possible_slope_values = all_slope_values - {val}
            for perturb_val in possible_slope_values:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                delta, is_pt = self._is_program_transition(
                    position, start_state, perturb_state, grad_m
                )
                if is_pt:
                    program_transitions.add(perturb_state)
                else:
                    if (
                        np.round(abs(perturb_val - val), self.slope_step_decimals)
                        <= self.slope_step
                    ):
                        deltas[position][perturb_state] = delta

        if not program_transitions:
            if start_state in self.invariants:
                # invariants might have no program transition and that is fine
                pass
            else:
                # deltas_abs = [abs(d) for d in deltas]
                # max_delta = max(deltas_abs)
                # max_delta_pos = [i for i, d in enumerate(deltas_abs) if d == max_delta]
                print("program transitions not found for", start_state)
                # print("max delta", max_delta, "max delta pos", max_delta_pos)
                print("deltas", deltas)
                input()

        # print(start_state, program_transitions)

        return program_transitions

    def _get_cvfs(self, start_state):
        cvfs = dict()
        all_slope_values = set(
            np.arange(self.min_slope, self.max_slope + self.slope_step, self.slope_step)
        )
        for position, val in enumerate(start_state):
            possible_slope_values = all_slope_values - {val}
            for perturb_val in possible_slope_values:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                cvfs[perturb_state] = (
                    position  # track the nodes to calculate its overall rank effect
                )

        return cvfs

    def __save_pts_to_file(self):
        def _map_key(state):
            return json.dumps([float(k) for k in state])

        pts = {
            _map_key(state): list(pts["program_transitions"])
            for state, pts in self.pts_n_cvfs.items()
        }

        json.dump(pts, open("output.json", "w"))
