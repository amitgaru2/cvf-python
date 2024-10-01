import os
import csv
import math
import json
import time
import random

import redis
import pandas as pd

from custom_logger import logger

FullAnalysisType = "full"
PartialAnalysisType = "partial"


class CVFAnalysis:
    graphs_dir = "graphs"
    results_dir = "results"
    analysis_type = "full"
    results_prefix = ""

    def __init__(self, graph_name, graph) -> None:
        self.graph_name = graph_name
        self.graph = graph
        self.nodes = list(graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}
        self.configurations = set()
        self.invariants = set()
        self.pts_rank = dict()
        self.pts_n_cvfs = dict()
        self.pts_rank_effect = dict()
        self.cvfs_in_rank_effect = dict()
        self.cvfs_out_rank_effect = dict()
        self.cvfs_in_rank_effect_df = None
        self.cvfs_out_rank_effect_df = None

        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)

        self.create_results_dir_if_not_exists()

    @staticmethod
    def gen_implicit_graph(no_nodes: int) -> dict:
        raise NotImplemented

    def start(self):
        start_time = time.time()
        self._start()
        logger.info("Total time taken: %s seconds.", round(time.time() - start_time, 2))

    def _start(self):
        self._gen_configurations()
        self._find_invariants()
        self._init_pts_rank()
        self._find_program_transitions_n_cvfs()
        self._rank_all_states()
        self._gen_save_rank_count()
        self._calculate_pts_rank_effect()
        self._calculate_cvfs_rank_effect()
        self._gen_save_rank_effect_count()
        self._gen_save_rank_effect_by_node_count()

    def create_results_dir_if_not_exists(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _gen_configurations(self):
        raise NotImplemented

    def _find_invariants(self):
        raise NotImplemented

    def _add_to_invariants(self, state):
        self.invariants.add(state)
        self.pts_rank[state] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}

    def _init_pts_rank(self):
        for inv in self.invariants:
            self.pts_rank[inv] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        raise NotImplemented

    def _get_program_transitions(self, start_state):
        raise NotImplemented

    def _get_cvfs(self, start_state):
        raise NotImplemented

    def _find_program_transitions_n_cvfs(self):
        logger.info("Finding Program Transitions and CVFS.")
        for state in self.configurations:
            self.pts_n_cvfs[state] = {
                "program_transitions": self._get_program_transitions(state),
            }
            key = "cvfs_in" if state in self.invariants else "cvfs_out"
            self.pts_n_cvfs[state][key] = self._get_cvfs(state)

    def _rank_all_states(self):
        total_paths = 0
        total_computation_paths = 0
        logger.info("Ranking all states .")
        unranked_states = set(self.pts_n_cvfs.keys()) - set(self.pts_rank.keys())
        logger.info("No. of Unranked states: %s", len(unranked_states))

        count = 0
        # rank the states that has all the paths to the ranked one
        while unranked_states:
            ranked_states = set(self.pts_rank.keys())
            remove_from_unranked_states = set()
            for state in unranked_states:
                dests = self.pts_n_cvfs[state]["program_transitions"]
                if (
                    dests - ranked_states
                ):  # some desitnations states are yet to be ranked
                    pass
                else:  # all the destination has been ranked
                    total_path_length = 0
                    path_count = 0
                    _max = 0
                    for succ in dests:
                        path_count += self.pts_rank[succ]["C"]
                        total_path_length += (
                            self.pts_rank[succ]["L"] + self.pts_rank[succ]["C"]
                        )
                        _max = max(_max, self.pts_rank[succ]["M"])
                        total_computation_paths += 1
                    self.pts_rank[state] = {
                        "L": total_path_length,
                        "C": path_count,
                        "A": total_path_length / path_count,
                        "Ar": math.ceil(total_path_length / path_count),
                        "M": _max + 1,
                    }
                    total_paths += path_count
                    remove_from_unranked_states.add(state)

            if not remove_from_unranked_states:
                count += 1
                if count % 10 == 0:
                    json.dump(list(unranked_states), open("unranked_states.json", "w"))
                    logger.error("Failed to rank states within 10 iterations.")
                    exit(1)
            else:
                count = 0
            unranked_states -= remove_from_unranked_states
            logger.debug("No. of Unranked states: %s", len(unranked_states))

        print("Total paths:", total_paths)
        print("Total computation paths:", total_computation_paths)

    def _calculate_pts_rank_effect(self):
        logger.info("Calculating Program Transition rank effect.")
        for state, pt_cvfs in self.pts_n_cvfs.items():
            for pt in pt_cvfs["program_transitions"]:
                self.pts_rank_effect[(state, pt)] = {
                    "Ar": self.pts_rank[pt]["Ar"] - self.pts_rank[state]["Ar"],
                    "M": self.pts_rank[pt]["M"] - self.pts_rank[state]["M"],
                }

    def _calculate_cvfs_rank_effect(self):
        logger.info("Calculating CVF rank effect.")
        for state, pt_cvfs in self.pts_n_cvfs.items():
            if "cvfs_in" in pt_cvfs:
                for cvf, node in pt_cvfs["cvfs_in"].items():
                    self.cvfs_in_rank_effect[(state, cvf)] = {
                        "node": node,
                        "Ar": self.pts_rank[cvf]["Ar"] - self.pts_rank[state]["Ar"],
                        "M": self.pts_rank[cvf]["M"] - self.pts_rank[state]["M"],
                    }
            else:
                for cvf, node in pt_cvfs["cvfs_out"].items():
                    self.cvfs_out_rank_effect[(state, cvf)] = {
                        "node": node,
                        "Ar": self.pts_rank[cvf]["Ar"] - self.pts_rank[state]["Ar"],
                        "M": self.pts_rank[cvf]["M"] - self.pts_rank[state]["M"],
                    }

    def _gen_save_rank_count(self):
        pt_rank_ = []
        for state in self.pts_rank:
            pt_rank_.append({"state": state, **self.pts_rank[state]})

        pt_rank_df = pd.DataFrame(pt_rank_)

        pt_avg_counts = pt_rank_df["Ar"].value_counts()
        pt_max_counts = pt_rank_df["M"].value_counts()

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

    def _gen_save_rank_effect_count(self):
        pt_rank_effect_ = []
        for state in self.pts_rank_effect:
            pt_rank_effect_.append({"state": state, **self.pts_rank_effect[state]})

        pt_rank_effect_df = pd.DataFrame(pt_rank_effect_)

        cvfs_in_rank_effect_ = []
        for state in self.cvfs_in_rank_effect:
            cvfs_in_rank_effect_.append(
                {"state": state, **self.cvfs_in_rank_effect[state]}
            )

        self.cvfs_in_rank_effect_df = pd.DataFrame(cvfs_in_rank_effect_)

        cvfs_out_rank_effect_ = []
        for state in self.cvfs_out_rank_effect:
            cvfs_out_rank_effect_.append(
                {"state": state, **self.cvfs_out_rank_effect[state]}
            )

        self.cvfs_out_rank_effect_df = pd.DataFrame(cvfs_out_rank_effect_)

        pt_avg_counts = pt_rank_effect_df["Ar"].value_counts()
        pt_max_counts = pt_rank_effect_df["M"].value_counts()
        cvf_in_avg_counts = self.cvfs_in_rank_effect_df["Ar"].value_counts()
        cvf_in_max_counts = self.cvfs_in_rank_effect_df["M"].value_counts()
        cvf_out_avg_counts = self.cvfs_out_rank_effect_df["Ar"].value_counts()
        cvf_out_max_counts = self.cvfs_out_rank_effect_df["M"].value_counts()

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
        cvf_in_avg_counts_by_node = self.cvfs_in_rank_effect_df.groupby(["node", "Ar"])[
            "Ar"
        ].count()
        cvf_in_max_counts_by_node = self.cvfs_in_rank_effect_df.groupby(["node", "M"])[
            "M"
        ].count()
        cvf_out_avg_counts_by_node = self.cvfs_out_rank_effect_df.groupby(
            ["node", "Ar"]
        )["Ar"].count()
        cvf_out_max_counts_by_node = self.cvfs_out_rank_effect_df.groupby(
            ["node", "M"]
        )["M"].count()

        max_Ar = max(
            self.cvfs_in_rank_effect_df["Ar"].max(),
            self.cvfs_out_rank_effect_df["Ar"].max(),
        )
        min_Ar = min(
            self.cvfs_in_rank_effect_df["Ar"].min(),
            self.cvfs_out_rank_effect_df["Ar"].min(),
        )

        max_M = max(
            self.cvfs_in_rank_effect_df["M"].max(),
            self.cvfs_out_rank_effect_df["M"].max(),
        )
        min_M = min(
            self.cvfs_in_rank_effect_df["M"].min(),
            self.cvfs_out_rank_effect_df["M"].min(),
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
                    node_re = (node, rank_effect)
                    writer.writerow(
                        {
                            "Node": node,
                            "Rank Effect": rank_effect,
                            "CVF In (Max)": cvf_in_max_counts_by_node.get(node_re, 0),
                            "CVF In (Avg)": cvf_in_avg_counts_by_node.get(node_re, 0),
                            "CVF Out (Max)": cvf_out_max_counts_by_node.get(node_re, 0),
                            "CVF Out (Avg)": cvf_out_avg_counts_by_node.get(node_re, 0),
                        }
                    )


class PartialCVFAnalysisMixin:
    DEFAULT_SAMPLE_SIZE = 10000

    @property
    def sample_size(self):
        return getattr(self, "_sample_size", self.DEFAULT_SAMPLE_SIZE)

    @sample_size.setter
    def sample_size(self, val: int):
        self._sample_size = val

    @property
    def analysis_type(self):
        return f"partial_{self.sample_size}"

    def _find_rank_of_successors(self, state, probe_limit, init=False):
        if state in self.pts_rank:
            return self.pts_rank[state]
        else:
            successors = list(self._get_program_transitions(state))
            if state in self.pts_rank:  # update due to LR
                return self.pts_rank[state]
            random.shuffle(successors)
            share = probe_limit // len(successors)
            add_extra_to_nodes = probe_limit - share * len(successors)
            total_path_length = 0
            path_count = 0
            _max = 0
            for succ in successors:
                if add_extra_to_nodes > 0:
                    result = self._find_rank_of_successors(succ, share + 1)
                    add_extra_to_nodes -= 1
                else:
                    if share > 0:
                        result = self._find_rank_of_successors(succ, share)
                    else:
                        break

                self.pts_n_cvfs[state]["program_transitions"].add(succ)

                path_count += result["C"]
                total_path_length += result["L"] + result["C"]
                _max = max(_max, result["M"])

            self.pts_rank[state] = {  # caching the rank for traveresed config
                "L": total_path_length,
                "C": path_count,
                "A": total_path_length / path_count,
                "Ar": math.ceil(total_path_length / path_count),
                "M": _max + 1,
            }

            return {
                "L": total_path_length,
                "C": path_count,
                "A": total_path_length / path_count,
                "Ar": math.ceil(total_path_length / path_count),
                "M": _max + 1,
            }

    def _find_program_transitions_n_cvfs(self):
        logger.info(
            "Finding Program Transitions and CVFS. Sample size: %d", self.sample_size
        )
        for state in self.configurations:
            self.pts_n_cvfs[state] = {"program_transitions": set()}

        for state in self.configurations:
            self._find_rank_of_successors(state, self.sample_size, True)

        for state in self.configurations:
            key = "cvfs_in" if state in self.invariants else "cvfs_out"
            self.pts_n_cvfs[state][key] = self._get_cvfs(state)

    # def _start(self):
    #     self._gen_configurations()
    #     self._find_invariants()
    #     self._init_pts_rank()
    #     self._find_program_transitions_n_cvfs()  # this does ranking as well
    #     # self._rank_all_states()
    #     self._gen_save_rank_count()
    #     self._calculate_pts_rank_effect()
    #     self._calculate_cvfs_rank_effect()
    #     self._gen_save_rank_effect_count()
    #     self._gen_save_rank_effect_by_node_count()
