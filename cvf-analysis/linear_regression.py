import itertools
import os
import copy
import json

import numpy as np

from mpi4py import MPI

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger

from lr_configs.config_adapter import LRConfig


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class LinearRegressionFullAnalysis(CVFAnalysis):

    @property
    def results_dir(self):
        return os.path.join("results", "linear_regression")

    @property
    def results_prefix(self):
        return f"linear_regression__{self.config.min_slope}_{self.config.max_slope}__{self.config.slope_step}__{self.config.matrix_id}"

    def __init__(self, graph_name, graph, config_file) -> None:
        super().__init__(graph_name, graph)
        self.cache = {"p": {}, "q": {}, "r": {}}
        self.config = LRConfig.generate_config(config_file)
        self.nodes = list(range(self.config.no_of_nodes))

        self.configurations_id = dict()
        self.configurations_count = 0

    # def __gen_test_data_partition_frm_df(self, partitions, df):
    #     shuffled = df.sample(frac=1)
    #     result = np.array_split(shuffled, partitions)
    #     return result

    def _start(self):
        self._gen_configurations()
        # self._find_program_transitions_n_cvfs()
        # # self._init_pts_rank()
        # # self.__save_pts_to_file()
        # self._rank_all_states()
        # self._gen_save_rank_count()
        # self._calculate_pts_rank_effect()
        # self._calculate_cvfs_rank_effect()
        # self._gen_save_rank_effect_count()
        # self._gen_save_rank_effect_by_node_count()

    def _gen_configurations(self):
        logger.debug("Generating configurations...")
        # if rank == 0:
        #     self.configurations = {
        #         tuple([self.config.min_slope for _ in range(self.config.no_of_nodes)])
        #     }
        # self.configurations = comm.bcast(self.configurations, root=0)
        possible_values = np.round(
            np.arange(
                self.config.min_slope,
                self.config.max_slope + self.config.slope_step,
                self.config.slope_step,
            ),
            self.config.slope_step_decimals,
        )
        config = [None for _ in range(self.config.no_of_nodes)]
        starting_values_from_rows = []
        offset = 0
        while True:
            if len(possible_values) > offset + rank:
                starting_values_from_rows.append(offset + rank)
                offset += comm.size
            else:
                break

        for sv in starting_values_from_rows:
            config[0] = sv
            other_node_values = itertools.product(possible_values, repeat=self.config.no_of_nodes-1)
            for nv in other_node_values:
                config[1:] = nv[:]
                config_cpy = tuple(config)
                self.configurations_id[config_cpy] = f"{rank}#{self.configurations_count}"
                self.configurations.add(config_cpy)
                self.configurations_count += 1

        logger.info("No. of Configurations: %s", self.configurations_count)

    def _find_invariants(self):
        logger.info("No. of Invariants: %s", len(self.invariants))

    # def __forward(self, X, params):
    #     return params["m"] * X + params["c"]

    # def __loss_fn(self, y, y_pred):
    #     N = len(y)
    #     return (1 / N) * sum((y[i] - y_pred[i]) ** 2 for i in range(N))

    def __get_f(self, state, node_id):
        doubly_st_mt = self.config.doubly_stochastic_matrix[node_id]
        return sum(frac * state[j] for j, frac in enumerate(doubly_st_mt))

    def __get_p(self, node_id):
        if node_id in self.cache["p"]:
            return self.cache["p"][node_id]

        df = self.__get_node_data_df(node_id)
        N = len(df)
        result = -2 / N
        self.cache["p"][node_id] = result
        return result

    def __get_q(self, node_id):
        if node_id in self.cache["q"]:
            return self.cache["q"][node_id]

        df = self.__get_node_data_df(node_id)
        result = np.sum(df["Xy"])
        self.cache["q"][node_id] = result
        return result

    def __get_r(self, node_id):
        if node_id in self.cache["r"]:
            return self.cache["r"][node_id]

        df = self.__get_node_data_df(node_id)
        result = np.sum(df["X_2"])
        self.cache["r"][node_id] = result
        return result

    # def __gradient_m(self, X, y, y_pred):
    #     N = len(y)
    #     return (-2 / N) * np.sum(X * (y - y_pred))

    def __get_node_data_df(self, node_id):
        # return self.config.df[self.config.df["node"] == node_id]
        return self.config.df[self.config.df["node"] == 1]

    # def __get_node_test_data_df(self, node_id):
    #     return self.df[self.df["node"] == node_id]

    def __clean_float_to_step_size_single(self, slope):
        quotient = np.divide(slope, self.config.slope_step)
        if quotient == int(quotient):
            return np.round(slope, self.config.slope_step_decimals)
        return np.round(
            np.int64(quotient) * self.config.slope_step, self.config.slope_step_decimals
        )

    def __copy_replace_indx_value(self, lst, indx, value):
        lst_copy = lst.copy()
        lst_copy[indx] = value
        return lst_copy

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        node_params = list(start_state)

        for node_id in range(self.config.no_of_nodes):
            for i in range(1, self.config.iterations + 1):
                prev_m = node_params[node_id]

                start_state_cpy = list(start_state)
                start_state_cpy[node_id] = prev_m

                # node_df = self.__get_node_data_df(node_id)
                # X_node = node_df["X"].array
                # y_node = node_df["y"].array

                # y_node_pred = self.__forward(X_node, {"m": prev_m, "c": 0})
                # grad_m = self.__gradient_m(X_node, y_node, y_node_pred)

                # doubly_st_mt = self.doubly_stochastic_matrix_config[node_id]

                # new_slope = (
                #     sum(
                #         frac * start_state_cpy[j]
                #         for j, frac in enumerate(doubly_st_mt)
                #     )
                #     - self.learning_rate * grad_m
                # )

                new_slope = self.__get_f(
                    start_state_cpy, node_id
                ) - self.config.learning_rate * self.__get_p(node_id) * (
                    self.__get_q(node_id) - prev_m * self.__get_r(node_id)
                )

                # if new_slope < self.config.min_slope:
                #     new_slope = self.config.min_slope

                if new_slope > self.config.max_slope:
                    new_slope = self.config.max_slope

                node_params[node_id] = new_slope

                if abs(prev_m - new_slope) <= self.config.stop_threshold:
                    break
            else:
                logger.debug(
                    "Couldn't converge node %s for the state %s", node_id, start_state
                )

        for node_id, new_slope in enumerate(node_params):
            new_slope_cleaned = self.__clean_float_to_step_size_single(new_slope)
            if new_slope_cleaned != start_state[node_id]:
                new_node_params = self.__copy_replace_indx_value(
                    list(start_state), node_id, new_slope_cleaned
                )
                new_node_params = tuple(new_node_params)
                program_transitions.add(new_node_params)

        if not program_transitions:
            self._add_to_invariants(start_state)
            logger.debug("No program transition found for %s !", start_state)

        return program_transitions

    def _get_cvfs(self, start_state):
        cvfs = dict()
        all_slope_values = set(
            np.round(
                np.arange(
                    self.config.min_slope,
                    self.config.max_slope + self.config.slope_step,
                    self.config.slope_step,
                ),
                self.config.slope_step_decimals,
            )
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


class LinearRegressionPartialAnalysis(
    PartialCVFAnalysisMixin, LinearRegressionFullAnalysis
):
    pass
