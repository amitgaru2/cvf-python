import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


COLORING_PROGRAM = "coloring"
DIJKSTRA_PROGRAM = "dijkstra_token_ring"
MAX_MATCHING_PROGRAM = "maximal_matching"
MAX_INDEPENDENT_SET_PROGRAM = "maximal_independent_set"
LINEAR_REGRESSION_PROGRAM = "linear_regression"


results_dir = "results"
program = "dijkstra_token_ring"  # coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
program = sys.argv[1]
programs = {
    DIJKSTRA_PROGRAM,
    COLORING_PROGRAM,
    MAX_MATCHING_PROGRAM,
    MAX_INDEPENDENT_SET_PROGRAM,
    LINEAR_REGRESSION_PROGRAM,
}
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

fontsize = 20

graph_names_map = {
    COLORING_PROGRAM: {
        # "graph_1",
        # "graph_2",
        # "graph_3",
        # "graph_6",
        # "graph_6b",
        "graph_7",
    },
    DIJKSTRA_PROGRAM: {
        "implicit_graph_n5",
        # "implicit_graph_n10",
        # "implicit_graph_n11",
        # "implicit_graph_n12",
        # "implicit_graph_n13",
    },
    MAX_MATCHING_PROGRAM: {
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_6",
        "graph_6b",
    },
    MAX_INDEPENDENT_SET_PROGRAM: {
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_4",
        "graph_5",
        "graph_6",
        "graph_6b",
        "graph_7",
        "graph_8",
    },
    LINEAR_REGRESSION_PROGRAM: {
        # "0.8_1.9__0.025__test_lr_graph_1",
        # "0.8_1.9__0.025__2__test_lr_graph_1",
        # "0.8_1.9__0.025__1__test_lr_graph_1",
        # "0.8_1.9__0.025__2__test_lr_graph_1",
        # "1.3_1.9__0.025__matrix_4__test_lr_graph_1",
        # "1.3_1.9__0.025__matrix_7__test_lr_graph_1",
        "1.3_1.9__0.025__matrix_6__test_lr_graph_1",
        # "0.7_1.9__0.025__test_lr_graph_1",
        # "0.9_1.9__0.025__test_lr_graph_1",
    },
}

graph_names = graph_names_map[program]

plots_dir = os.path.join("plots", program)


def get_df(graph_name):
    full_path = os.path.join(
        results_dir,
        program,
        f"rank_effect_by_node__{analysis_type}__{program}__{graph_name}.csv",
    )
    if not os.path.exists(full_path):
        print("File not found:", full_path)
        return None

    df = pd.read_csv(full_path)
    df["CVF (Avg)"] = df["CVF In (Avg)"] + df["CVF Out (Avg)"]
    df["CVF (Max)"] = df["CVF In (Max)"] + df["CVF Out (Max)"]
    return df


def create_plots_dir_if_not_exists():
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


create_plots_dir_if_not_exists()
plt.figure(figsize=(16, 8))

markers = ["*", "o", "h", "v", "P", "s", "p", "x", "D", "8"]

for graph_name in graph_names:
    df = get_df(graph_name)
    if df is None:
        continue

    rank_effects = df["Rank Effect"].unique()
    rank_effects.sort()
    df_preproc = pd.DataFrame({"Rank Effect": rank_effects})
    nodes = df["Node"].unique()
    nodes.sort()
    for node in nodes:
        col = f"Node {node}"
        node_data = df.loc[(df["Node"] == node)]["CVF (Avg)"]
        node_data = node_data.reset_index(drop=True)
        df_preproc.loc[:, col] = node_data

    df_preproc.set_index("Rank Effect", inplace=True)

    ax = sns.lineplot(
        data=df_preproc[[f"Node {i}" for i in nodes]], linewidth=3, markersize=10
    )

    for i, line in enumerate(ax.lines):
        line.set_marker(markers[i % len(nodes)])

    ax.set_xlabel("Rank Effect")

    # Set custom ticks and labels
    # ax.set_xlim(-16, 16)
    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    file_name = f"rank_effect_by_node__{analysis_type}__{program}__{graph_name}"
  
    plt.rc("font", size=fontsize)
    plt.legend(fontsize=fontsize)  # using a size in points
    plt.savefig(
        os.path.join(
            plots_dir,
            f"{file_name}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plot(s) for {graph_name}.")
