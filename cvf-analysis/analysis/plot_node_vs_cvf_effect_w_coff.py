import os
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from plot_config import *


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
    return df


def create_plots_dir_if_not_exists():
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


def plot_node_vs_rank_effect(df, ax, node_id_max, c_off):
    if program == "coloring":
        sns.scatterplot(data=df, x="Node", y="Rank Effect", ax=ax, s=500)
    elif program == "maximal_matching":
        sns.scatterplot(data=df, x="Node", y="Rank Effect", ax=ax, s=250)
    else:
        sns.scatterplot(data=df, x="Node", y="Rank Effect", ax=ax)
    rank_effect_max = df["Rank Effect"].max()
    ax.set_xlabel("Node ID")
    ax.set_xlim(left=-0.5, right=node_id_max + 0.5)
    ax.set_xticks([i for i in range(0, node_id_max + 1)])
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    if program == "coloring" or program == "maximal_matching":
        ax.set_yticks([i for i in range(c_off, rank_effect_max + 1)])


if __name__ == "__main__":

    plots_dir = os.path.join("plots", program, "node_vs_cvf_effect")

    create_plots_dir_if_not_exists()

    for indx, graph_name in enumerate(graph_names):
        df = get_df(graph_name)
        if df is None:
            continue
        cut_off = graph_names[graph_name]["cut_off"]
        node_id_max = df.agg({"Node": ["max"]})["Node"]["max"]
        grps = df[(df["CVF (Avg)"] > 0) & (df["Rank Effect"] > cut_off)].groupby(
            ["Node", "Rank Effect"]
        )
        data = grps.groups.keys()
        df = pd.DataFrame(data, columns=["Node", "Rank Effect"])
        fig, ax = plt.subplots(1, figsize=(10, 5), constrained_layout=True)
        file_name = f"node__vs__rank_effect_gte_{cut_off}__{analysis_type}__{program}__{graph_name}"
        fig_title = f"Node vs Rank effect >= {cut_off} for program {program_label}"
        # fig.suptitle(fig_title, fontsize=fontsize)
        plot_node_vs_rank_effect(df, ax, node_id_max, cut_off)

        fig.savefig(
            os.path.join(
                plots_dir,
                f"{file_name}.png",
            )
        )

        print(f"Saved plot for {graph_name}")
