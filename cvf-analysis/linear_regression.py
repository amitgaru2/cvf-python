import os
import csv
import math
import time
import json
import pickle
import hashlib
import itertools

import redis
import numpy as np
import pandas as pd

from functools import reduce
from collections import Counter

from custom_mpi import comm, program_node_rank
from lr_configs.config_adapter import LRConfig
from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


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

        # redis related configurations
        self.config_id_key_prefix = "cid"
        self.config_rank_key_prefix = "cr"
        self.configs_ranked_key_prefix = "crd"
        self.redis_client = redis.StrictRedis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379,
            username="default",
            password=os.getenv("REDIS_PASSWORD", ""),
            db=0,
        )
        if program_node_rank == 0:
            self.redis_client.flushdb()
        comm.barrier()

        self.configurations_id = dict()
        self.possible_values = np.round(
            np.arange(
                self.config.min_slope,
                self.config.max_slope + self.config.slope_step,
                self.config.slope_step,
            ),
            self.config.slope_step_decimals,
        )

    # def __gen_test_data_partition_frm_df(self, partitions, df):
    #     shuffled = df.sample(frac=1)
    #     result = np.array_split(shuffled, partitions)
    #     return result

    def _start(self):
        self._gen_configurations()
        comm.barrier()
        self._find_program_transitions_n_cvfs()
        comm.barrier()
        # # self._init_pts_rank()
        # # self.__save_pts_to_file()
        self._rank_all_states()
        comm.barrier()
        self._gen_save_rank_count()
        comm.barrier()
        self._calculate_pts_rank_effect()
        comm.barrier()
        self._calculate_cvfs_rank_effect()
        comm.barrier()
        if program_node_rank == 0:
            self._gen_save_rank_effect_count()
            self._gen_save_rank_effect_by_node_count()
        comm.barrier()
        # self._gen_save_rank_effect_by_node_count()

    def get_config_dump(self, config):
        return pickle.dumps(config)

    def hash_config(self, config):
        return hashlib.md5(str(config).encode()).hexdigest()

    def _gen_configurations(self):
        logger.debug("Generating configurations...")
        config = [None for _ in range(self.config.no_of_nodes)]
        starting_values_from_rows = []
        offset = 0
        while True:
            if len(self.possible_values) > offset + program_node_rank:
                starting_values_from_rows.append(offset + program_node_rank)
                offset += comm.size
            else:
                break

        configuration_count = 0
        for sv in starting_values_from_rows:
            config[0] = self.possible_values[sv]
            other_node_values = itertools.product(
                self.possible_values, repeat=self.config.no_of_nodes - 1
            )
            for nv in other_node_values:
                config[1:] = nv[:]
                config_cpy = tuple(config)
                config_hash = self.hash_config(config_cpy)
                self.set_data_to_redis(
                    f"{self.config_id_key_prefix}_{config_hash}",
                    f"{program_node_rank}#{configuration_count}",
                )
                # self.set_data_to_redis(
                #     f"config_tuple__{config_hash}", self.get_config_dump(config_cpy)
                # )
                # self.set_data_to_redis(
                #     f"config_hash__{program_node_rank}#{configuration_count}",
                #     config_hash,
                # )
                self.configurations.add(config_cpy)
                configuration_count += 1

        logger.info("No. of Configurations: %s", len(self.configurations))

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

    def save_rank(self, state_id, rank):
        self.set_data_to_redis(
            f"{self.config_rank_key_prefix}_{state_id}", json.dumps(rank)
        )
        self.sadd_data_to_redis(self.configs_ranked_key_prefix, state_id)

    def set_data_to_redis(self, key, value, backoff=True):
        logger.debug('Setting key "%s" in redis.', key)
        result = self.redis_client.set(key, value)
        if not result:
            logger.error('Failed to set key "%s" in redis.', key)
            exit(1)

    def sadd_data_to_redis(self, key, value, backoff=True):
        logger.debug('Setting set key "%s" in redis.', key)
        result = self.redis_client.sadd(key, value)
        if not result:
            logger.error('Failed to set key "%s" in redis.', key)
            exit(1)

    def get_data_frm_redis(self, key, backoff=True):
        result = self.redis_client.get(key)
        if result is None and backoff:
            for i in range(5):
                sleep_for = 2**i
                logger.debug(
                    'Redis data not found. Sleeping for "%s" seconds.', sleep_for
                )
                time.sleep(sleep_for)
                result = self.redis_client.get(key)
                if result is not None:
                    break
            else:
                logger.error('Redis data not found for key "%s".', key)
                exit(1)
        return result

    def get_rank_data_key_index(self, key):
        return {"L": 0, "C": 1, "A": 2, "Ar": 0, "M": 0}[key]

    def _add_to_invariants(self, state):
        self.invariants.add(state)
        state_id = self.get_data_frm_redis(
            f"{self.config_id_key_prefix}_{self.hash_config(state)}"
        ).decode()
        # {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0} => [0, 1, 0, 0, 0]
        self.save_rank(state_id, [0, 1, 0, 0, 0])
        # self.pts_rank[state_id] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}

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
            for _ in range(1, self.config.iterations + 1):
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
                config_id = self.redis_client.get(
                    f"{self.config_id_key_prefix}_{self.hash_config(new_node_params)}"
                ).decode()
                if config_id is None:
                    logger.error(
                        "Config id not found for the config %s", new_node_params
                    )
                    exit(1)
                program_transitions.add(config_id)

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
                config_id = self.get_data_frm_redis(
                    f"{self.config_id_key_prefix}_{self.hash_config(perturb_state)}"
                ).decode()
                cvfs[config_id] = (
                    position  # track the nodes to calculate its overall rank effect
                )

        return cvfs

    def _find_program_transitions_n_cvfs(self):
        logger.info("Finding Program Transitions and CVFS.")
        for state in self.configurations:
            state_id = self.get_data_frm_redis(
                f"{self.config_id_key_prefix}_{self.hash_config(state)}"
            ).decode()
            self.pts_n_cvfs[state_id] = {
                "program_transitions": self._get_program_transitions(state),
            }
            key = "cvfs_in" if state in self.invariants else "cvfs_out"
            self.pts_n_cvfs[state_id][key] = self._get_cvfs(state)

    def _rank_all_states(self):
        total_paths = 0
        total_computation_paths = 0
        logger.info("Ranking all states .")
        ranked_states = {
            c.decode()
            for c in self.redis_client.smembers(self.configs_ranked_key_prefix)
        }
        unranked_states = set(self.pts_n_cvfs.keys()) - set(ranked_states)
        logger.info("No. of Unranked states: %s", len(unranked_states))

        count = 0
        # rank the states that has all the paths to the ranked one
        while unranked_states:
            # ranked_states = set(self.pts_rank.keys())
            ranked_states = {
                c.decode()
                for c in self.redis_client.smembers(self.configs_ranked_key_prefix)
            }
            remove_from_unranked_states = set()
            for state_id in unranked_states:
                dests = self.pts_n_cvfs[state_id]["program_transitions"]
                if (
                    dests - ranked_states
                ):  # some desitnations states are yet to be ranked
                    pass
                else:  # all the destination has been ranked
                    total_path_length = 0
                    path_count = 0
                    _max = 0
                    for succ in dests:
                        # path_count += self.pts_rank[succ]["C"]
                        pts_rank_succ = json.loads(
                            self.get_data_frm_redis(
                                f"{self.config_rank_key_prefix}_{succ}"
                            )
                        )
                        path_count += pts_rank_succ["C"]
                        total_path_length += pts_rank_succ["L"] + pts_rank_succ["C"]
                        _max = max(_max, pts_rank_succ["M"])
                        total_computation_paths += 1
                    # self.pts_rank[state] = {
                    #     "L": total_path_length,
                    #     "C": path_count,
                    #     "A": total_path_length / path_count,
                    #     "Ar": math.ceil(total_path_length / path_count),
                    #     "M": _max + 1,
                    # }
                    self.save_rank(
                        state_id,
                        {
                            "L": total_path_length,
                            "C": path_count,
                            "A": total_path_length / path_count,
                            "Ar": math.ceil(total_path_length / path_count),
                            "M": _max + 1,
                        },
                    )
                    total_paths += path_count
                    remove_from_unranked_states.add(state_id)

            if not remove_from_unranked_states:
                count += 1
                if count % 100 == 0:
                    json.dump(list(unranked_states), open("unranked_states.json", "w"))
                    logger.error("Failed to rank states within 10 iterations.")
                    exit(1)
            else:
                count = 0
            unranked_states -= remove_from_unranked_states
            logger.debug("No. of Unranked states: %s", len(unranked_states))

        logger.debug("Total paths: %s", total_paths)
        logger.debug("Total computation paths: %s", total_computation_paths)

    def _reduce_pt_counts_df(self, pt_counts: list):
        return reduce(
            lambda left, right: left.add(right, fill_value=0), pt_counts
        ).astype(int)

    def _gen_save_rank_count(self):
        pt_rank_ = []
        for state_id in self.pts_n_cvfs:
            state_pts_rank = json.loads(
                self.get_data_frm_redis(f"{self.config_rank_key_prefix}_{state_id}")
            )
            pt_rank_.append({**state_pts_rank})

        pt_rank_df = pd.DataFrame(pt_rank_)

        pt_avg_counts = pt_rank_df["Ar"].value_counts()
        pt_max_counts = pt_rank_df["M"].value_counts()

        pt_avg_counts = comm.gather(pt_avg_counts, root=0)
        pt_max_counts = comm.gather(pt_max_counts, root=0)

        if program_node_rank == 0:
            pt_avg_counts = self._reduce_pt_counts_df(pt_avg_counts)
            pt_max_counts = self._reduce_pt_counts_df(pt_max_counts)

            fieldnames = ["Rank", "Count (Max)", "Count (Avg)"]
            with open(
                os.path.join(
                    self.results_dir,
                    f"rank__{self.analysis_type}__{self.results_prefix}__{self.graph_name}.csv",
                ),
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for rank in sorted(set(pt_avg_counts.index) | set(pt_max_counts.index)):
                    writer.writerow(
                        {
                            "Rank": rank,
                            "Count (Max)": pt_max_counts.get(rank, 0),
                            "Count (Avg)": pt_avg_counts.get(rank, 0),
                        }
                    )

    def _reduce_pts_rank_effect(self, rank_effects):
        result = {}
        result["Ar"] = reduce(
            lambda left, right: {"Ar": left["Ar"].add(right["Ar"], fill_value=0)},
            rank_effects,
        )["Ar"]
        result["M"] = reduce(
            lambda left, right: {"M": left["M"].add(right["M"], fill_value=0)},
            rank_effects,
        )["M"]
        return pd.DataFrame(result).fillna(0).astype("int")

    def _reduce_cvfs_rank_effect(self, rank_effects):
        result = {}
        for node in self.nodes:
            result[node] = {}
            result[node]["Ar"] = reduce(
                lambda left, right: (
                    {
                        node: {
                            "Ar": (
                                left[node]["Ar"].add(right[node]["Ar"], fill_value=0)
                                if node in left and node in right
                                else (
                                    left[node]["Ar"]
                                    if node in left
                                    else (
                                        right[node]["Ar"]
                                        if node in right
                                        else pd.Series([])
                                    )
                                )
                            )
                        }
                    }
                ),
                rank_effects,
            )[node]["Ar"].astype(int)
            result[node]["M"] = reduce(
                lambda left, right: (
                    {
                        node: {
                            "M": (
                                left[node]["M"].add(right[node]["M"], fill_value=0)
                                if node in left and node in right
                                else (
                                    left[node]["M"]
                                    if node in left
                                    else (
                                        right[node]["M"]
                                        if node in right
                                        else pd.Series([])
                                    )
                                )
                            )
                        }
                    }
                ),
                rank_effects,
            )[node]["M"].astype(int)

        result = pd.DataFrame.from_dict(result, orient="index")
        return result

    def _calculate_pts_rank_effect(self):
        logger.info("Calculating Program Transition rank effect.")
        Ar = []
        M = []
        self.pts_rank_effect = {"Ar": Ar, "M": M}
        for state, pt_cvfs in self.pts_n_cvfs.items():
            state_pts_rank = json.loads(
                self.get_data_frm_redis(f"{self.config_rank_key_prefix}_{state}")
            )
            for pt in pt_cvfs["program_transitions"]:
                pt_pts_rank = json.loads(
                    self.get_data_frm_redis(f"{self.config_rank_key_prefix}_{pt}")
                )
                Ar.append(pt_pts_rank["Ar"] - state_pts_rank["Ar"])
                M.append(pt_pts_rank["M"] - state_pts_rank["M"])

        # locally reduce
        self.pts_rank_effect["Ar"] = pd.Series(Counter(self.pts_rank_effect["Ar"]))
        self.pts_rank_effect["M"] = pd.Series(Counter(self.pts_rank_effect["M"]))

        self.pts_rank_effect = comm.gather(self.pts_rank_effect, root=0)

        if program_node_rank == 0:
            self.pts_rank_effect = self._reduce_pts_rank_effect(self.pts_rank_effect)
            self.pts_rank_effect.to_csv("pts_rank_effect.csv")

    def _calculate_cvfs_rank_effect(self):
        logger.info("Calculating CVF rank effect.")
        for state, pt_cvfs in self.pts_n_cvfs.items():
            state_pts_rank = json.loads(
                self.get_data_frm_redis(f"{self.config_rank_key_prefix}_{state}")
            )
            if "cvfs_in" in pt_cvfs:
                for cvf, node in pt_cvfs["cvfs_in"].items():
                    cvf_pts_rank = json.loads(
                        self.get_data_frm_redis(f"{self.config_rank_key_prefix}_{cvf}")
                    )
                    if node not in self.cvfs_in_rank_effect:
                        self.cvfs_in_rank_effect[node] = {
                            "Ar": [cvf_pts_rank["Ar"] - state_pts_rank["Ar"]],
                            "M": [cvf_pts_rank["M"] - state_pts_rank["M"]],
                        }
                    else:
                        self.cvfs_in_rank_effect[node]["Ar"].append(
                            cvf_pts_rank["Ar"] - state_pts_rank["Ar"]
                        )
                        self.cvfs_in_rank_effect[node]["M"].append(
                            cvf_pts_rank["M"] - state_pts_rank["M"]
                        )
            else:
                for cvf, node in pt_cvfs["cvfs_out"].items():
                    cvf_pts_rank = json.loads(
                        self.get_data_frm_redis(f"{self.config_rank_key_prefix}_{cvf}")
                    )
                    if node not in self.cvfs_out_rank_effect:
                        self.cvfs_out_rank_effect[node] = {
                            "Ar": [cvf_pts_rank["Ar"] - state_pts_rank["Ar"]],
                            "M": [cvf_pts_rank["M"] - state_pts_rank["M"]],
                        }
                    else:
                        self.cvfs_out_rank_effect[node]["Ar"].append(
                            cvf_pts_rank["Ar"] - state_pts_rank["Ar"]
                        )
                        self.cvfs_out_rank_effect[node]["M"].append(
                            cvf_pts_rank["M"] - state_pts_rank["M"]
                        )

        # locally reduce
        for node, rank_effects in self.cvfs_in_rank_effect.items():
            rank_effects["Ar"] = pd.Series(Counter(rank_effects["Ar"]))
            rank_effects["M"] = pd.Series(Counter(rank_effects["M"]))

        for node, rank_effects in self.cvfs_out_rank_effect.items():
            rank_effects["Ar"] = pd.Series(Counter(rank_effects["Ar"]))
            rank_effects["M"] = pd.Series(Counter(rank_effects["M"]))

        self.cvfs_in_rank_effect = comm.gather(self.cvfs_in_rank_effect, root=0)
        self.cvfs_out_rank_effect = comm.gather(self.cvfs_out_rank_effect, root=0)

        if program_node_rank == 0:
            self.cvfs_in_rank_effect = self._reduce_cvfs_rank_effect(
                self.cvfs_in_rank_effect
            )
            self.cvfs_out_rank_effect = self._reduce_cvfs_rank_effect(
                self.cvfs_out_rank_effect
            )
            # self.cvfs_in_rank_effect.to_csv("cvfs_in_rank_effect.csv")

    def _gen_save_rank_effect_count(self):
        pt_avg_counts = self.pts_rank_effect["Ar"]
        pt_max_counts = self.pts_rank_effect["M"]
        cvf_in_avg_counts = reduce(
            lambda i, j: i.add(j, fill_value=0),
            self.cvfs_in_rank_effect[:]["Ar"],
        ).astype(int)
        cvf_in_max_counts = reduce(
            lambda i, j: i.add(j, fill_value=0),
            self.cvfs_in_rank_effect[:]["M"],
        ).astype(int)
        cvf_out_avg_counts = reduce(
            lambda i, j: i.add(j, fill_value=0),
            self.cvfs_out_rank_effect[:]["Ar"],
        ).astype(int)
        cvf_out_max_counts = reduce(
            lambda i, j: i.add(j, fill_value=0),
            self.cvfs_out_rank_effect[:]["M"],
        ).astype(int)

        fieldnames = [
            "Rank Effect",
            "PT (Max)",
            "PT (Avg)",
            "CVF In (Max)",
            "CVF In (Avg)",
            "CVF Out (Max)",
            "CVF Out (Avg)",
        ]
        with open(
            os.path.join(
                self.results_dir,
                f"rank_effect__{self.analysis_type}__{self.results_prefix}__{self.graph_name}.csv",
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for re in sorted(
                set(pt_avg_counts.index)
                | set(pt_max_counts.index)
                | set(cvf_in_avg_counts.index)
                | set(cvf_in_max_counts.index)
                | set(cvf_out_avg_counts.index)
                | set(cvf_out_max_counts.index)
            ):
                writer.writerow(
                    {
                        "Rank Effect": re,
                        "PT (Max)": pt_max_counts.get(re, 0),
                        "PT (Avg)": pt_avg_counts.get(re, 0),
                        "CVF In (Max)": cvf_in_max_counts.get(re, 0),
                        "CVF In (Avg)": cvf_in_avg_counts.get(re, 0),
                        "CVF Out (Max)": cvf_out_max_counts.get(re, 0),
                        "CVF Out (Avg)": cvf_out_avg_counts.get(re, 0),
                    }
                )

    def _gen_save_rank_effect_by_node_count(self):
        max_Ar = max(
            reduce(
                lambda i, j: max(i, max(j.index)), self.cvfs_in_rank_effect[:]["Ar"], 0
            ),
            reduce(
                lambda i, j: max(i, max(j.index)), self.cvfs_out_rank_effect[:]["Ar"], 0
            ),
        )
        min_Ar = min(
            reduce(
                lambda i, j: min(i, min(j.index)), self.cvfs_in_rank_effect[:]["Ar"], 0
            ),
            reduce(
                lambda i, j: min(i, min(j.index)), self.cvfs_out_rank_effect[:]["Ar"], 0
            ),
        )

        max_M = max(
            reduce(
                lambda i, j: max(i, max(j.index)), self.cvfs_in_rank_effect[:]["M"], 0
            ),
            reduce(
                lambda i, j: max(i, max(j.index)), self.cvfs_out_rank_effect[:]["M"], 0
            ),
        )
        min_M = min(
            reduce(
                lambda i, j: min(i, min(j.index)), self.cvfs_in_rank_effect[:]["M"], 0
            ),
            reduce(
                lambda i, j: min(i, min(j.index)), self.cvfs_out_rank_effect[:]["M"], 0
            ),
        )

        max_Ar_M = max(max_Ar, max_M)
        min_Ar_M = min(min_Ar, min_M)

        # rank effect of individual node
        fieldnames = [
            "Node",
            "Rank Effect",
            "CVF In (Max)",
            "CVF In (Avg)",
            "CVF Out (Max)",
            "CVF Out (Avg)",
        ]
        with open(
            os.path.join(
                self.results_dir,
                f"rank_effect_by_node__{self.analysis_type}__{self.results_prefix}__{self.graph_name}.csv",
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for node in self.nodes:
                for rank_effect in range(min_Ar_M, max_Ar_M + 1):
                    # node_re = (node, rank_effect)
                    writer.writerow(
                        {
                            "Node": node,
                            "Rank Effect": rank_effect,
                            "CVF In (Max)": self.cvfs_in_rank_effect.iloc[node][
                                "M"
                            ].get(rank_effect, 0),
                            "CVF In (Avg)": self.cvfs_in_rank_effect.iloc[node][
                                "Ar"
                            ].get(rank_effect, 0),
                            "CVF Out (Max)": self.cvfs_out_rank_effect.iloc[node][
                                "M"
                            ].get(rank_effect, 0),
                            "CVF Out (Avg)": self.cvfs_out_rank_effect.iloc[node][
                                "Ar"
                            ].get(rank_effect, 0),
                        }
                    )

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
