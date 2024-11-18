import os
import sys

COLORING_PROGRAM = "coloring"
DIJKSTRA_PROGRAM = "dijkstra_token_ring"
MAX_MATCHING_PROGRAM = "maximal_matching"
MAX_INDEPENDENT_SET_PROGRAM = "maximal_independent_set"
LINEAR_REGRESSION_PROGRAM = "linear_regression"

results_dir = os.path.join(os.pardir, "results")

programs = {
    DIJKSTRA_PROGRAM,
    COLORING_PROGRAM,
    MAX_MATCHING_PROGRAM,
    MAX_INDEPENDENT_SET_PROGRAM,
    LINEAR_REGRESSION_PROGRAM
}
# coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
program = sys.argv[1]
if program not in programs:
    print(f"Program {program} not found.")
    exit(1)

program_label_map = {
    "dijkstra_token_ring": "dijkstra",
    "coloring": "graph coloring",
    "maximal_matching": "maximal matching",
    "maximal_independent_set": "maximal indp. set",
    "linear_regression": "linear regression",
}
program_label = program_label_map.get(program, program)

analysis_type = "full"  # full, partial

fontsize = 30

graph_names_map = {
    COLORING_PROGRAM: {
        "graph_1": {"cut_off": 0},
        "graph_2": {"cut_off": 0},
        "graph_3": {"cut_off": 0},
        "graph_6": {"cut_off": 0},
        "graph_6b": {"cut_off": 0},
        "graph_7": {"cut_off": 0},
    },
    DIJKSTRA_PROGRAM: {
        "implicit_graph_n10": {"cut_off": 40},
        "implicit_graph_n11": {"cut_off": 40},
        "implicit_graph_n12": {"cut_off": 50},
        "implicit_graph_n13": {"cut_off": 60},
    },
    MAX_MATCHING_PROGRAM: {
        "graph_1": {"cut_off": 20},
        "graph_2": {"cut_off": 10},
        "graph_3": {"cut_off": 15},
        "graph_6": {"cut_off": 10},
        "graph_6b": {"cut_off": 10},
    },
    MAX_INDEPENDENT_SET_PROGRAM: {
        "graph_1": {"cut_off": 0},
        "graph_2": {"cut_off": 0},
        "graph_3": {"cut_off": 0},
        "graph_4": {"cut_off": 0},
        "graph_5": {"cut_off": 0},
        "graph_6": {"cut_off": 0},
        "graph_6b": {"cut_off": 0},
        "graph_7": {"cut_off": 0},
        "graph_8": {"cut_off": 0},
    },
    LINEAR_REGRESSION_PROGRAM: {
        "0.8_1.9__0.025__test_lr_graph_1": {"cut_off": 0}
    }
}


graph_names = graph_names_map[program]
